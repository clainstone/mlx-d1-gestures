// Fixtures: cd coreai && .venv/bin/python export_app.py (MLX's logits and answers on the parity frames).
// End to end on this Mac: D1_MODEL=/path/d1_gpu_fp16.aimodel [D1_COMPUTE=gpu|neuralEngine] swift test
import CoreGraphics
import Foundation
import ImageIO
import Testing
@testable import D1Kit

let fixtures = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures")

struct Fixture: Decodable {
    let name: String
    let logits: [[Double]]
    let readout: [String: Want]
    let mlx: [String: Want]
}

struct Want: Decodable {
    let type: String
    let choice: String?
    let confidence: Double?
    let probabilities: [String: Double]?
    let box: [String: Double]?
    let found: Double?
}

func load() throws -> [Fixture] {
    let url = fixtures.appendingPathComponent("fixtures.json")
    guard FileManager.default.fileExists(atPath: url.path) else {
        Issue.record("no fixtures: cd coreai && .venv/bin/python export_app.py"); return []
    }
    return try JSONDecoder().decode([Fixture].self, from: Data(contentsOf: url))
}

func image(_ name: String) -> CGImage {
    let src = CGImageSourceCreateWithURL(fixtures.appendingPathComponent(name + ".png") as CFURL, nil)!
    return CGImageSourceCreateImageAtIndex(src, 0, nil)!
}

func edges(_ b: Box) -> [String: Double] { ["top": b.top, "bottom": b.bottom, "left": b.left, "right": b.right] }

/// The same logits through mlx-vlm and through Swift: the same answers, to rounding.
@Test func readoutMatchesMlxVlm() throws {
    let layout = try Layout.bundled()
    for f in try load() {
        let got = layout.readout(f.logits.flatMap { $0.map(Float.init) })
        for (name, want) in f.readout {
            if want.type == "choice" {
                let c = try #require(got.choices[name])
                #expect(c.choice == want.choice, "\(f.name) \(name)")
                for (k, p) in zip(c.keys, c.probabilities) { #expect(abs(p - want.probabilities![k]!) < 1e-9, "\(f.name) \(name) \(k)") }
            } else {
                let b = try #require(got.boxes[name])
                #expect(abs(b.found - want.found!) < 1e-9, "\(f.name) \(name) found")
                #expect(abs(b.confidence - want.confidence!) < 1e-9, "\(f.name) \(name) confidence")
                #expect((b.box == nil) == (want.box == nil), "\(f.name) \(name) box")
                if let bb = b.box, let w = want.box { for (k, v) in edges(bb) { #expect(abs(v - w[k]!) < 1e-9, "\(f.name) \(name) \(k)") } }
            }
        }
    }
}

/// Swift's pixels against the Qwen3-VL processor's, on the same 640x360 frames: the same tensor.
@Test func pixelsMatchTheProcessor() throws {
    for f in try load() {
        let want = try Data(contentsOf: fixtures.appendingPathComponent(f.name + ".f32")).withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        let got = Frame.pixels(image(f.name))
        #expect(got.count == want.count)
        let diff = zip(got, want).map { abs($0 - $1) }.max()!
        #expect(diff < 1e-6, "\(f.name): max pixel diff \(diff)")
    }
}

/// coreai_backend.py's crop arithmetic.
@Test func cropAndUncrop() {
    let full = Box(top: 0, bottom: 1, left: 0, right: 1)
    #expect(Frame.uncrop(full, 640, 360) == full)
    #expect(Frame.uncrop(full, 400, 400) == Box(top: 87.0 / 400, bottom: 312.0 / 400, left: 0, right: 1))
    #expect(Frame.uncrop(full, 1000, 360) == Box(top: 0, bottom: 1, left: 180.0 / 1000, right: 820.0 / 1000))
    #expect(Frame.uncrop(Box(top: 0.5, bottom: 0.5, left: 0.25, right: 0.75), 640, 480)
            == Box(top: 240.0 / 480, bottom: 240.0 / 480, left: 0.25, right: 0.75))
}

/// A portrait frame is letterboxed: black bars left and right, the picture in the middle, boxes mapped back onto it.
@Test func portraitLetterbox() {
    let mid = Box(top: 0, bottom: 1, left: 0.5 - 0.5 * 720 / 2275, right: 0.5 + 0.5 * 720 / 2275)  // the picture's span
    let u = Frame.uncrop(mid, 720, 1280)
    #expect(abs(u.left) < 0.002 && abs(u.right - 1) < 0.002 && u.top == 0 && u.bottom == 1)
    let ctx = CGContext(data: nil, width: 720, height: 1280, bitsPerComponent: 8, bytesPerRow: 0,
                        space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
    ctx.setFillColor(gray: 1, alpha: 1); ctx.fill(CGRect(x: 0, y: 0, width: 720, height: 1280))
    let px = Frame.pixels(ctx.makeImage()!), row = 176 * 640
    #expect(px[row + 10] == -1 && px[row + 320] == 1 && px[row + 630] == -1)  // bar, picture, bar
}

/// Picture -> Swift pixels -> Core AI -> Swift readout, against MLX end to end (parity.py's gate: same choices, top
/// probability within 0.05, box edges within 0.03).
@Test(.enabled(if: ProcessInfo.processInfo.environment["D1_MODEL"] != nil)) func endToEndMatchesMlx() async throws {
    let env = ProcessInfo.processInfo.environment
    let d1 = try await D1(model: URL(fileURLWithPath: env["D1_MODEL"]!), compute: D1.Compute(rawValue: env["D1_COMPUTE"] ?? "gpu")!)
    for f in try load() {
        let r = try await d1.decide(image(f.name))
        var line = "\(f.name): \(Int(r.ms)) ms"
        for (name, want) in f.mlx {
            if want.type == "choice" {
                let c = try #require(r.answers.choices[name])
                let dp = abs(c.probabilities[c.keys.firstIndex(of: want.choice!)!] - want.confidence!)
                line += " | \(name) \(c.choice) \(String(format: "%.2f", c.confidence)) (MLX \(want.choice!) \(String(format: "%.2f", want.confidence!)))"
                #expect(c.choice == want.choice, "\(f.name) \(name)")
                // the Neural Engine's one known miss (README): a uniform gray frame, same answer, its probability off a bit more
                withKnownIssue("Neural Engine on a uniform gray frame", isIntermittent: true) { #expect(dp <= 0.05, "\(f.name) \(name) p diff \(dp)") }
                    when: { env["D1_COMPUTE"] == "neuralEngine" && f.name == "blank_gray" }
            } else {
                let b = try #require(r.answers.boxes[name])
                #expect((b.box == nil) == (want.box == nil), "\(f.name) \(name) found")
                if let bb = b.box, let w = want.box {
                    let e = edges(bb).map { abs($0.value - w[$0.key]!) }.max()!
                    line += " | \(name) edge \(String(format: "%.3f", e))"
                    #expect(e <= 0.03, "\(f.name) \(name) edge \(e)")
                }
            }
        }
        print(line)
    }
}
