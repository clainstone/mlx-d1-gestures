// d1 gestures on iPhone and iPad: the front camera in, hand gesture and facial expression out, with d1-2B (Liquid AI)
// on Core AI, the Neural Engine or the GPU. Everything stays on the device.
import D1Kit
import SwiftUI

@main
struct D1GesturesApp: App {
    var body: some Scene {
        WindowGroup { ContentView().preferredColorScheme(.dark) }
    }
}

// The labels for the keys of QUESTIONS (decide.py, coreai/questions.py), as on the web page.
let HAND = ["thumbs_up": "👍 Thumbs up", "thumbs_down": "👎 Thumbs down", "ok": "👌 OK", "peace": "✌️ Peace / Victory",
            "open_palm": "✋ Open palm", "pointing": "☝️ Pointing", "fist": "✊ Fist", "none": "· No gesture"]
let FACE = ["smile": "🙂 Smile", "sad": "🙁 Sad", "surprised": "😮 Surprised", "neutral": "😐 Neutral"]

@MainActor @Observable
final class Live {
    var compute: D1.Compute = ProcessInfo.processInfo.environment["D1_COMPUTE"] == "gpu" ? .gpu : .neuralEngine  // env: debugging
    var status = "Starting…" {
        didSet { if ProcessInfo.processInfo.environment["D1_TIMING"] != nil, !status.hasPrefix("Downloading") { print("d1 status: \(status)") } }
    }
    var progress: Double?  // the download, 0...1, while it runs
    var preparing = false  // Core AI specialising the model
    var cameraLive = false
    var decision: Decision?
    var size = CGSize(width: 16, height: 9)  // the decided frame's, for its boxes
    var rate = 0.0
    var inputTokens = 0
    var off: Set<String> = []  // "hand" / "face" switched off with a tap: no box, no bars
    let camera = Camera()
    private var d1: D1?
    private var loop: Task<Void, Never>?

    func start() async {
        UIApplication.shared.isIdleTimerDisabled = true  // the download and the first start both need d1 open
        await load()
    }

    /// The camera opens only once a model is ready to read it.
    private func openCamera() async -> Bool {
        if cameraLive { return true }
        do { try await camera.start() } catch { status = error.localizedDescription; return false }
        cameraLive = true
        return true
    }

    func select(_ c: D1.Compute) async {
        guard c != compute || d1 == nil else { return }
        compute = c
        loop?.cancel(); loop = nil
        d1 = nil; decision = nil  // one model in memory at a time
        await load()
    }

    private func load() async {
        let c = compute
        do {
            var url = Models.local(c)
            if !Models.ready(c) {
                progress = 0
                status = "Downloading the \(c.label) model (once; keep d1 open)"
                defer { progress = nil }
                url = try await Models.fetch(c) { got, total in
                    Task { @MainActor in
                        guard total > 0 else { return }
                        self.progress = Double(got) / Double(total)
                        self.status = String(format: "Downloading the %@ model: %.1f of %.1f GB (once; keep d1 open)", c.label, Double(got) / 1e9, Double(total) / 1e9)
                    }
                }
            }
            guard c == compute else { return }
            status = c == .neuralEngine ? "Preparing the Neural Engine: about a minute the very first time, then a second…" : "Preparing the GPU…"
            preparing = true
            defer { preparing = false }
            let t = Date()
            let m = try await D1(model: url, compute: c)
            guard c == compute, await openCamera() else { return }
            d1 = m
            inputTokens = m.layout.inputTokens
            status = String(format: "%@ ready in %.1f s", c.label, Date().timeIntervalSince(t))
            run(m)
        } catch {
            status = "\(c.label): \(error.localizedDescription)"
        }
    }

    /// One frame in flight at a time: the next one is the latest the camera has when this answer comes back.
    private func run(_ m: D1) {
        loop = Task {
            var n = 0, t1 = Date()
            while !Task.isCancelled {
                guard let img = camera.latest else { try? await Task.sleep(for: .milliseconds(50)); continue }
                do {
                    let r = try await m.decide(img)
                    if Task.isCancelled { break }
                    decision = r
                    size = CGSize(width: img.width, height: img.height)
                    n += 1
                    let dt = Date().timeIntervalSince(t1)
                    if dt > 1 { rate = Double(n) / dt; n = 0; t1 = Date() }
                } catch {
                    status = "\(compute.label): \(error.localizedDescription)"
                    try? await Task.sleep(for: .seconds(1.5))
                }
            }
        }
    }
}

extension D1.Compute {
    var label: String { self == .gpu ? "GPU" : "Neural Engine" }
}

struct ContentView: View {
    @State private var live = Live()

    var body: some View {
        GeometryReader { g in
            if g.size.height > g.size.width {  // portrait: the camera on top, the rest under it
                VStack(spacing: 12) {
                    Stage(live: live, portrait: true).frame(maxHeight: g.size.height * 0.5)
                    Stats(live: live)
                    ScrollView { Side(live: live) }
                }
            } else {
                HStack(alignment: .top, spacing: 20) {
                    VStack(spacing: 12) {
                        Stage(live: live, portrait: false)
                        Stats(live: live)
                    }
                    ScrollView { Side(live: live) }.frame(width: 300)
                }
            }
        }
        .padding(16)
        .background(Color(red: 0.05, green: 0.07, blue: 0.09))
        .task { await live.start() }
    }
}

/// The mirrored camera with the hand and face boxes on it (the boxes mirror with the picture, their labels don't).
/// In portrait the frame is 9:16 and d1 sees it letterboxed (Frame.crop16x9).
struct Stage: View {
    let live: Live
    let portrait: Bool

    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 18).fill(Color.white.opacity(0.04))
            if live.cameraLive {
                CameraPreview(layer: live.camera.preview)
                    .overlay { GeometryReader { g in boxes(g.size) }.scaleEffect(x: -1) }  // the preview layer mirrors itself
                    .clipShape(RoundedRectangle(cornerRadius: 18))
            } else {
                VStack(spacing: 14) {
                    if let p = live.progress {
                        ProgressView(value: p).tint(.orange).frame(maxWidth: 360)
                        Text("\(Int(p * 100))%").font(.title3.monospacedDigit().weight(.semibold))
                    } else if live.preparing {
                        ProgressView().controlSize(.large)
                    }
                    Text(live.status).foregroundStyle(.secondary).multilineTextAlignment(.center)
                }
                .padding(24)
            }
        }
        .aspectRatio(portrait ? 9 / 16 : 16 / 9, contentMode: .fit)
        .frame(maxWidth: .infinity)
    }

    @ViewBuilder func boxes(_ s: CGSize) -> some View {
        if let a = live.decision?.answers {
            ForEach([("hand_box", "hand", HAND, Color.orange), ("face_box", "face", FACE, Color.teal)], id: \.0) { key, q, labels, color in
                if !live.off.contains(q), let b = a.boxes[key]?.box, let c = a.choices[q], c.choice != "none" {
                    let u = Frame.uncrop(b, Int(live.size.width), Int(live.size.height))
                    let r = CGRect(x: u.left * s.width, y: u.top * s.height, width: (u.right - u.left) * s.width, height: (u.bottom - u.top) * s.height)
                    RoundedRectangle(cornerRadius: 10).stroke(color, lineWidth: 3)
                        .frame(width: r.width, height: r.height)
                        .overlay(alignment: .topLeading) {
                            Text("\(labels[c.choice] ?? c.choice) \(Int(c.confidence * 100))%")
                                .font(.caption.monospaced().weight(.semibold)).foregroundStyle(.black)
                                .padding(.horizontal, 6).padding(.vertical, 2).background(color, in: .rect(cornerRadius: 6))
                                .scaleEffect(x: -1).fixedSize().offset(y: r.minY < 24 ? 0 : -22)
                        }
                        .position(x: r.midX, y: r.midY)
                        .animation(.linear(duration: 0.12), value: r)
                }
            }
        }
    }
}

struct Side: View {
    let live: Live

    var body: some View {
        let a = live.decision?.answers
        VStack(alignment: .leading, spacing: 14) {
            HStack(spacing: 10) {
                headline("hand", a?.choices["hand"], HAND)
                headline("face", a?.choices["face"], FACE)
            }
            if !live.off.contains("hand") {
                bars(a?.choices["hand"], HAND, order: ["thumbs_up", "thumbs_down", "ok", "peace", "open_palm", "pointing", "fist", "none"])
            }
            if !live.off.contains("face") { bars(a?.choices["face"], FACE, order: ["smile", "sad", "surprised", "neutral"]) }
            Picker("Backend", selection: Binding(get: { live.compute }, set: { c in Task { await live.select(c) } })) {
                ForEach(D1.Compute.allCases, id: \.self) { Text("Core AI " + $0.label).tag($0) }
            }
            .pickerStyle(.segmented)
            Text(live.status).font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            Text("d1-2B by Liquid AI, on Core AI. Frames never leave this device.").font(.caption2).foregroundStyle(.tertiary)
        }
    }

    /// Tap to switch the question's box and bars off (and on again).
    func headline(_ q: String, _ c: ChoiceAnswer?, _ labels: [String: String]) -> some View {
        let off = live.off.contains(q)
        return Button {
            if off { live.off.remove(q) } else { live.off.insert(q) }
        } label: {
            VStack(alignment: .leading, spacing: 2) {
                Text(q.uppercased()).font(.caption2.monospaced()).foregroundStyle(.secondary)
                Text(off ? "off" : c.map { $0.choice == "none" ? "—" : "\(labels[$0.choice] ?? $0.choice) \(Int($0.confidence * 100))%" } ?? "—")
                    .font(.headline).lineLimit(1).minimumScaleFactor(0.6)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(10).background(Color.white.opacity(0.05), in: .rect(cornerRadius: 12))
            .opacity(off ? 0.4 : 1)
        }
        .buttonStyle(.plain)
        .accessibilityLabel("\(q), \(off ? "off" : "on")")
        .accessibilityHint("Shows or hides its box and its decision")
    }

    func bars(_ c: ChoiceAnswer?, _ labels: [String: String], order: [String]) -> some View {
        VStack(spacing: 3) {
            ForEach(order, id: \.self) { k in
                let p = c.flatMap { c in c.keys.firstIndex(of: k).map { c.probabilities[$0] } } ?? 0
                let top = c?.choice == k
                HStack(spacing: 8) {
                    Text(labels[k] ?? k).font(.footnote).foregroundStyle(top ? .primary : .secondary).frame(width: 130, alignment: .leading).lineLimit(1)
                    ProgressView(value: p).tint(top ? .orange : .gray)
                    Text(c == nil ? "—" : "\(Int(p * 100))%").font(.footnote.monospacedDigit()).frame(width: 40, alignment: .trailing)
                }
                .animation(.linear(duration: 0.15), value: p)
            }
        }
    }

}

/// Under the camera: how fast d1 decides, what it reads, and where it runs.
struct Stats: View {
    let live: Live

    var body: some View {
        Grid(horizontalSpacing: 10, verticalSpacing: 10) {
            GridRow {
                tile("ms per decision", live.decision.map { "\(Int($0.ms.rounded()))" })
                tile("decisions / s", live.rate > 0 ? String(format: "%.1f", live.rate) : nil)
            }
            GridRow {
                tile("input tokens", live.inputTokens > 0 ? "\(live.inputTokens)" : nil)
                tile("running on", live.decision == nil ? nil : live.compute.label)
            }
        }
    }

    func tile(_ name: String, _ value: String?) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(name.uppercased()).font(.caption2.monospaced()).foregroundStyle(.secondary).lineLimit(1).minimumScaleFactor(0.7)
            Text(value ?? "—").font(.title2.monospacedDigit().weight(.semibold)).lineLimit(1).minimumScaleFactor(0.5)
                .contentTransition(.numericText())
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, 12).padding(.vertical, 8)
        .background(Color.white.opacity(0.05), in: .rect(cornerRadius: 12))
    }
}
