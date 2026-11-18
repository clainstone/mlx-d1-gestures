// d1-2B's decision pass on Core AI: a picture and the questions in, their answers out (one forward pass, no tokens).
// The questions are the bundled prompt inputs (prompt/*.f16, coreai/export_app.py), not part of the model.
import CoreAI
import CoreGraphics
import Foundation

public struct Decision: Sendable {
    public let answers: Answers
    public let ms: Double  // the whole decision: pixels, model, readout
}

public actor D1 {
    public enum Compute: String, Sendable, CaseIterable { case neuralEngine, gpu }

    public nonisolated let layout: Layout
    let fns: [InferenceFunction]  // one, or the pieces of a split export, run in order
    var input: NDArray
    var prompt: [String: NDArray] = [:]

    /// Specialises `model` (an .aimodel folder, or a folder of part0.aimodel ... partN.aimodel: the Neural Engine export in
    /// pieces, coreai/export_ane_split.py) for `compute`. The first time ever on a device this compiles it, which
    /// takes about a minute on the Neural Engine; Core AI caches the result, so later starts take about a second.
    public init(model: URL, compute: Compute) async throws {
        layout = try Layout.bundled()
        let kind: ComputeUnitKind = compute == .gpu ? .gpu : .neuralEngine
        let options = SpecializationOptions(preferredComputeUnitKind: kind)
        if D1.timing { print("d1 compute units: available \(ComputeUnitKind.availableKinds), allowed \(options.allowedComputeUnitKinds), preferred \(String(describing: options.preferredComputeUnitKind))") }
        var parts = [model]
        if model.pathExtension != "aimodel" {
            parts = try FileManager.default.contentsOfDirectory(at: model, includingPropertiesForKeys: nil)
                .filter { $0.pathExtension == "aimodel" }.sorted { $0.lastPathComponent.localizedStandardCompare($1.lastPathComponent) == .orderedAscending }
            guard parts.count > 1 else { throw D1Error("\(model.lastPathComponent) holds no model pieces") }
        }
        var fs: [InferenceFunction] = []
        for p in parts {
            fs.append(try await D1.function(p, options))
        }
        fns = fs
        input = NDArray(shape: [1, 3, layout.height, layout.width], scalarType: .float32)
        for (name, shape) in layout.prompt {
            let url = Bundle.module.url(forResource: name, withExtension: "f16", subdirectory: "prompt")!
            let data = try Data(contentsOf: url)
            guard data.count == shape.reduce(1, *) * 2 else { throw D1Error("prompt/\(name).f16 is not \(shape) fp16") }
            var a = NDArray(shape: shape, scalarType: .float16)
            var v = a.mutableView(as: Float16.self)
            v.withUnsafeMutablePointer { ptr, _, _ in data.withUnsafeBytes { src in _ = memcpy(ptr, src.baseAddress!, data.count) } }
            prompt[name] = a
        }
        do { _ = try await logits() } catch {  // the first run pays the lazy set-up
            if D1.timing { print("d1 first run failed: \(String(reflecting: error))") }
            throw D1Error("the first run failed: \(error.localizedDescription)")
        }
    }

    /// One piece's "main", specialised. A cached compile can go bad (iOS purges caches under storage pressure, which can
    /// leave an entry that no longer loads: a nil error from loadFunction); then its entry is deleted and it is compiled again, once.
    static func function(_ p: URL, _ options: SpecializationOptions) async throws -> InferenceFunction {
        for attempt in 0..<2 {
            do {
                let m = try await AIModel.specialize(contentsOf: p, options: options, cache: .default, cachePolicy: .default)
                guard let f = try m.loadFunction(named: "main") else { throw D1Error("\(p.lastPathComponent) has no main function") }
                return f
            } catch let e as D1Error { throw e } catch {
                if D1.timing { print("d1 \(p.lastPathComponent) attempt \(attempt) failed: \(String(reflecting: error))") }
                if attempt == 1 { throw D1Error("could not load \(p.lastPathComponent): \(error.localizedDescription)") }
                try? AIModelCache.default.deleteEntries(for: p)
            }
        }
        fatalError("unreachable")
    }

    /// D1_TIMING=1 in the environment: a per-stage breakdown on stdout every 30 decisions.
    static let timing = ProcessInfo.processInfo.environment["D1_TIMING"] != nil
    var stages = [Double](repeating: 0, count: 5), count = 0

    public func decide(_ image: CGImage) async throws -> Decision {
        let t = Date()
        var mark = t
        func lap(_ i: Int) { let now = Date(); stages[i] += now.timeIntervalSince(mark) * 1000; mark = now }
        let px = Frame.pixels(image, width: layout.width, height: layout.height)
        lap(0)
        var view = input.mutableView(as: Float.self)
        view.withUnsafeMutablePointer { ptr, _, _ in px.withUnsafeBufferPointer { ptr.update(from: $0.baseAddress!, count: px.count) } }
        let out = try await run()
        lap(1)
        let l = rows(out)
        lap(2)
        let answers = layout.readout(l)
        lap(3)
        let ms = Date().timeIntervalSince(t) * 1000
        if D1.timing {
            stages[4] += ms; count += 1
            if count == 30 {
                let a = stages.map { String(format: "%.1f", $0 / 30) }
                print("d1 timing (ms, mean of 30): pixels \(a[0]) | model \(a[1]) | flatten \(a[2]) | readout \(a[3]) | total \(a[4]) | thermal \(ProcessInfo.processInfo.thermalState.rawValue)")
                stages = stages.map { _ in 0 }; count = 0
            }
        }
        return Decision(answers: answers, ms: ms)
    }

    /// The function's logits as row-major [rows x columns].
    /// One forward pass: the logits NDArray. Pieces: the first takes the picture and the text, each passes the LM stream
    /// "x" on, the last reads out (export_ane_split.py's inputs).
    func run() async throws -> NDArray {
        func take(_ o: inout InferenceFunction.Outputs, _ name: String) throws -> NDArray {
            guard let a = o.remove(name)?.ndArray else { throw D1Error("the model gave no \(name)") }
            return a
        }
        let p = prompt
        if fns.count == 1 {
            var o = try await fns[0].run(inputs: p.merging(["image": input]) { $1 })
            return try take(&o, "logits")
        }
        var o = try await fns[0].run(inputs: ["image": input, "text": p["text"]!, "attend": p["attend"]!])
        var x = try take(&o, "x")
        for f in fns[1..<(fns.count - 1)] {
            var m = try await f.run(inputs: ["x": x, "attend": p["attend"]!])
            x = try take(&m, "x")
        }
        var last = try await fns[fns.count - 1].run(inputs: ["x": x, "attend": p["attend"]!, "slots": p["slots"]!, "head": p["head"]!])
        return try take(&last, "logits")
    }

    func logits() async throws -> [Float] { rows(try await run()) }

    func rows(_ out: NDArray) -> [Float] {
        let flat = flatten(out)
        guard out.shape.count == 4 else { return flat }
        let u = out.shape[1], s = out.shape[3]  // the Neural Engine layout, BC1S [1, U, 1, slots] -> [slots, U]
        var tr = [Float](repeating: 0, count: flat.count)
        for i in 0..<u { for j in 0..<s { tr[j * u + i] = flat[i * s + j] } }
        return tr
    }
}

public struct D1Error: LocalizedError {
    public let errorDescription: String?
    init(_ s: String) { errorDescription = s }
}

/// Stride-aware copy of a float16 / float32 NDArray to [Float], row-major (as coreai/swift's worker).
func flatten(_ array: NDArray) -> [Float] {
    func go<T: BinaryFloatingPoint & BitwiseCopyable>(_ t: T.Type) -> [Float] {
        let total = array.shape.reduce(1, *)
        var out = [Float](repeating: 0, count: total)
        array.view(as: t).withUnsafePointer { ptr, shape, strides in
            let rank = shape.count
            var idx = [Int](repeating: 0, count: rank)
            for i in 0..<total {
                var off = 0
                for d in 0..<rank { off += idx[d] * strides[d] }
                out[i] = Float(ptr[off])
                var d = rank - 1
                while d >= 0 { idx[d] += 1; if idx[d] < shape[d] { break }; idx[d] = 0; d -= 1 }
            }
        }
        return out
    }
    switch array.scalarType {
    case .float16: return go(Float16.self)
    case .float32: return go(Float.self)
    default: fatalError("unsupported output type \(array.scalarType)")
    }
}
