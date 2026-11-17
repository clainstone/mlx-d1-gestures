// The front camera at 1280x720 (16:9, the shape the export was traced at), upright. The preview is the capture
// session's own layer (no per-frame work); d1 gets the latest frame as a CGImage only when it asks for one.
import AVFoundation
import CoreImage
import SwiftUI

final class Camera: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate, @unchecked Sendable {
    let preview = AVCaptureVideoPreviewLayer()
    private let session = AVCaptureSession()
    private let queue = DispatchQueue(label: "camera")
    private let ci = CIContext()
    private let lock = NSLock()
    private var buffer: CVPixelBuffer?
    private var rotation: AVCaptureDevice.RotationCoordinator?
    private var watch: [NSKeyValueObservation] = []

    /// The latest frame, converted now (once per decision, not once per camera frame).
    var latest: CGImage? {
        guard let pb = lock.withLock({ buffer }) else { return nil }
        let image = CIImage(cvPixelBuffer: pb)
        return ci.createCGImage(image, from: image.extent)
    }

    func start() async throws {
        guard await AVCaptureDevice.requestAccess(for: .video) else {
            throw CameraError("Camera access is off: allow it in Settings → Privacy & Security → Camera.")
        }
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front) else {
            throw CameraError("No front camera.")
        }
        session.beginConfiguration()
        session.sessionPreset = .hd1280x720
        session.addInput(try AVCaptureDeviceInput(device: device))
        let out = AVCaptureVideoDataOutput()
        out.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        out.alwaysDiscardsLateVideoFrames = true
        out.setSampleBufferDelegate(self, queue: queue)
        session.addOutput(out)
        session.commitConfiguration()
        preview.session = session
        preview.videoGravity = .resizeAspectFill
        // frames and preview upright whichever way the device is turned (the app runs in landscape)
        let rc = AVCaptureDevice.RotationCoordinator(device: device, previewLayer: preview)
        let frames = out.connection(with: .video), shown = preview.connection
        frames?.videoRotationAngle = rc.videoRotationAngleForHorizonLevelCapture
        shown?.videoRotationAngle = rc.videoRotationAngleForHorizonLevelPreview
        watch = [
            rc.observe(\.videoRotationAngleForHorizonLevelCapture) { [queue] rc, _ in
                let angle = rc.videoRotationAngleForHorizonLevelCapture
                queue.async { frames?.videoRotationAngle = angle }
            },
            rc.observe(\.videoRotationAngleForHorizonLevelPreview) { rc, _ in
                let angle = rc.videoRotationAngleForHorizonLevelPreview
                DispatchQueue.main.async { shown?.videoRotationAngle = angle }
            },
        ]
        rotation = rc
        queue.async { self.session.startRunning() }
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sample: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pb = CMSampleBufferGetImageBuffer(sample) else { return }
        lock.withLock { buffer = pb }  // ponytail: holds one buffer of the capture pool; copy it if frames ever stall
    }
}

/// The capture session's preview layer in SwiftUI (mirrored for the front camera by AVFoundation itself).
struct CameraPreview: UIViewRepresentable {
    let layer: AVCaptureVideoPreviewLayer

    final class View: UIView {
        var preview: AVCaptureVideoPreviewLayer?
        override func layoutSubviews() { super.layoutSubviews(); preview?.frame = bounds }
    }

    func makeUIView(context: Context) -> View {
        let v = View()
        v.preview = layer
        v.layer.addSublayer(layer)
        return v
    }

    func updateUIView(_ v: View, context: Context) {}
}

struct CameraError: LocalizedError {
    let errorDescription: String?
    init(_ s: String) { errorDescription = s }
}
