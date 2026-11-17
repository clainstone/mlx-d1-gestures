// The two Core AI exports, downloaded from Hugging Face on first use into Application Support (not backed up). The
// Neural Engine one comes in 4 pieces (coreai/export_ane_split.py): iOS's Neural Engine compiler can't take 3.1 GB at once.
import D1Kit
import Foundation

enum Models {
    static let repo = "ivanfioravanti/d1-2B-gestures-CoreAI"
    static func folder(_ c: D1.Compute) -> String {
        if let f = ProcessInfo.processInfo.environment["D1_MODEL_FOLDER"] { return f }  // debugging: another export already in models/
        return c == .gpu ? "d1_gpu_prompt_fp16.aimodel" : "d1_ane_prompt_split_fp16"  // the questions are inputs (D1Kit prompt/)
    }

    /// The folder's files, each moved into place only once complete: one .aimodel's, or each piece's.
    static func files(_ c: D1.Compute) -> [String] {
        let one = ["metadata.json", "main.hash", "main.mlirb"]
        return folder(c).hasSuffix(".aimodel") ? one : (0..<4).flatMap { p in one.map { "part\(p).aimodel/\($0)" } }
    }

    static func local(_ c: D1.Compute) -> URL {
        URL.applicationSupportDirectory.appendingPathComponent("models").appendingPathComponent(folder(c))
    }

    static func ready(_ c: D1.Compute) -> Bool {
        files(c).allSatisfy { FileManager.default.fileExists(atPath: local(c).appendingPathComponent($0).path) }
    }

    /// The model folder, downloading what is missing; `progress(bytes, total)` over all of it (3.3 GB).
    static func fetch(_ c: D1.Compute, progress: @escaping @Sendable (Int64, Int64) -> Void) async throws -> URL {
        var dir = local(c)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        try dir.setResourceValues(values)
        let missing = files(c).filter { !FileManager.default.fileExists(atPath: dir.appendingPathComponent($0).path) }
        let sizes = try await Self.sizes(c)
        let total = missing.reduce(Int64(0)) { $0 + (sizes[$1] ?? 0) }
        var done: Int64 = 0
        for f in missing {
            let url = URL(string: "https://huggingface.co/\(repo)/resolve/main/\(folder(c))/\(f)")!
            let base = done
            let tmp = try await Download.run(url) { got, _ in progress(base + got, total) }
            let dst = dir.appendingPathComponent(f)
            try FileManager.default.createDirectory(at: dst.deletingLastPathComponent(), withIntermediateDirectories: true)
            try FileManager.default.moveItem(at: tmp, to: dst)
            done += sizes[f] ?? 0
        }
        return dir
    }

    /// Each file's size in the repo (for one progress bar over all of them), by its path inside the folder.
    static func sizes(_ c: D1.Compute) async throws -> [String: Int64] {
        struct Entry: Decodable { let path: String; let size: Int64? }
        let url = URL(string: "https://huggingface.co/api/models/\(repo)/tree/main/\(folder(c))?recursive=true")!
        let entries = try JSONDecoder().decode([Entry].self, from: try await URLSession.shared.data(from: url).0)
        return Dictionary(entries.map { (String($0.path.dropFirst(folder(c).count + 1)), $0.size ?? 0) }) { a, _ in a }
    }
}

/// One file over HTTPS with byte progress. ponytail: a foreground session, so keep the app open while it downloads;
/// a background URLSession if that gets in the way.
final class Download: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    private let progress: @Sendable (Int64, Int64) -> Void
    private var done: CheckedContinuation<URL, any Error>?
    private var file: URL?

    private init(_ progress: @escaping @Sendable (Int64, Int64) -> Void) { self.progress = progress }

    static func run(_ url: URL, progress: @escaping @Sendable (Int64, Int64) -> Void) async throws -> URL {
        let d = Download(progress)
        let session = URLSession(configuration: .default, delegate: d, delegateQueue: nil)
        defer { session.finishTasksAndInvalidate() }
        return try await withCheckedThrowingContinuation { c in
            d.done = c
            session.downloadTask(with: url).resume()
        }
    }

    func urlSession(_ s: URLSession, downloadTask: URLSessionDownloadTask, didWriteData: Int64, totalBytesWritten: Int64,
                    totalBytesExpectedToWrite: Int64) {
        progress(totalBytesWritten, totalBytesExpectedToWrite)
    }

    func urlSession(_ s: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {
        guard (downloadTask.response as? HTTPURLResponse)?.statusCode == 200 else { return }
        let keep = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        if (try? FileManager.default.moveItem(at: location, to: keep)) != nil { file = keep }  // `location` goes away on return
    }

    func urlSession(_ s: URLSession, task: URLSessionTask, didCompleteWithError error: (any Error)?) {
        if let file { done?.resume(returning: file) } else {
            let code = (task.response as? HTTPURLResponse)?.statusCode
            done?.resume(throwing: error ?? URLError(.badServerResponse, userInfo: [NSLocalizedDescriptionKey:
                "could not download \(task.originalRequest?.url?.lastPathComponent ?? "the model") (HTTP \(code.map(String.init) ?? "?"))"]))
        }
        done = nil
    }
}
