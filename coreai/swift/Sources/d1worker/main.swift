// d1worker: the exported d1 decision function on Core AI, as a JSON-lines worker.
//
//   d1worker --model out/d1_gpu_fp16.aimodel --compute gpu|ane
//
// stdin, one JSON object a line:
//   {"pixels": "/path/frame.f32"}                         -> {"logits": [12 x U floats, row-major], "ms": ...}
//   {"bench": "/path/frame.f32", "warmup": 5, "runs": 30} -> {"median_ms": ..., "min_ms": ..., "max_ms": ..., "runs_ms": [...]}
// A .f32 file is the raw float32 image tensor [1, 3, 352, 640] (d1_common.pixels). The first line on stdout reports
// the load: {"ready": true, "specialize_s": ..., "load_function_s": ..., "first_run_s": ...}.
import CoreAI
import Foundation

func emit(_ obj: [String: Any]) {
    let data = try! JSONSerialization.data(withJSONObject: obj)
    FileHandle.standardOutput.write(data + Data("\n".utf8))
}

func arg(_ name: String) -> String? {
    let a = CommandLine.arguments
    guard let i = a.firstIndex(of: name), i + 1 < a.count else { return nil }
    return a[i + 1]
}

func now() -> Double { Double(DispatchTime.now().uptimeNanoseconds) / 1e9 }

/// Stride-aware copy of any float16/float32 NDArray to [Float], row-major.
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

@main
struct Worker {
    static func main() async throws {
        guard let path = arg("--model") else { fputs("usage: d1worker --model X.aimodel --compute gpu|ane\n", stderr); exit(2) }
        let kind: ComputeUnitKind = (arg("--compute") ?? "gpu") == "ane" ? .neuralEngine : .gpu
        let url = URL(fileURLWithPath: path)

        var t = now()
        let model = try await AIModel.specialize(contentsOf: url, options: SpecializationOptions(preferredComputeUnitKind: kind),
                                                 cache: .default, cachePolicy: .default)
        let specialize = now() - t
        t = now()
        guard let fn = try model.loadFunction(named: "main") else { fatalError("no main function") }
        let loadFn = now() - t

        var input = NDArray(shape: [1, 3, 352, 640], scalarType: .float32)
        func fill(_ file: String) throws {
            let data = try Data(contentsOf: URL(fileURLWithPath: file))
            precondition(data.count == 3 * 352 * 640 * 4, "expected 3*352*640 float32")
            var view = input.mutableView(as: Float.self)
            view.withUnsafeMutablePointer { ptr, _, _ in
                data.withUnsafeBytes { src in _ = memcpy(ptr, src.baseAddress!, data.count) }
            }
        }
        func run() async throws -> [Float] {
            var outputs = try await fn.run(inputs: ["image": input])
            guard let out = outputs.remove("logits")?.ndArray else { fatalError("no logits output") }
            let flat = flatten(out)
            if out.shape.count == 4 {  // BC1S [1, U, 1, 12] -> [12, U]
                let u = out.shape[1], s = out.shape[3]
                var tr = [Float](repeating: 0, count: flat.count)
                for i in 0..<u { for j in 0..<s { tr[j * u + i] = flat[i * s + j] } }
                return tr
            }
            return flat
        }
        // first run on zeros: pays the lazy set-up
        t = now()
        _ = try await run()
        emit(["ready": true, "model": path, "compute": kind == .neuralEngine ? "ane" : "gpu",
              "specialize_s": specialize, "load_function_s": loadFn, "first_run_s": now() - t])

        while let line = readLine() {
            guard let req = try? JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any] else {
                emit(["error": "bad json"]); continue
            }
            do {
                if let file = req["pixels"] as? String {
                    try fill(file)
                    let t0 = now()
                    let logits = try await run()
                    emit(["logits": logits, "ms": (now() - t0) * 1000])
                } else if let file = req["bench"] as? String {
                    try fill(file)
                    for _ in 0..<(req["warmup"] as? Int ?? 5) { _ = try await run() }
                    var ms: [Double] = []
                    for _ in 0..<(req["runs"] as? Int ?? 30) {
                        let t0 = now(); _ = try await run(); ms.append((now() - t0) * 1000)
                    }
                    let s = ms.sorted()
                    emit(["median_ms": s[s.count / 2], "min_ms": s[0], "max_ms": s[s.count - 1], "runs_ms": ms])
                } else {
                    emit(["error": "expected pixels or bench"])
                }
            } catch {
                emit(["error": "\(error)"])
            }
        }
    }
}
