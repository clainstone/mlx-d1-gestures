// A picture as the exported function takes it, as decide.py makes it: centre-cropped to 16:9 (the shape the export was
// traced at) at 640x360 (the frame the page sends), resized to 640x352 by the Qwen3-VL processor (PIL bicubic), then
// [1, 3, 352, 640] float32 in [-1, 1].
import CoreGraphics

public enum Frame {
    /// The centre 16:9 rectangle of a w x h picture, in pixels (d1_common.crop16x9). A portrait picture (taller than
    /// wide) is letterboxed instead: the rectangle is wider than the picture, black on both sides, as a 16:9 crop of it
    /// would keep only a band across the middle.
    public static func crop16x9(_ w: Int, _ h: Int) -> CGRect {
        if h > w { let nw = h * 16 / 9; return CGRect(x: (w - nw) / 2, y: 0, width: nw, height: h) }
        if w * 9 > h * 16 { let nw = h * 16 / 9; return CGRect(x: (w - nw) / 2, y: 0, width: nw, height: h) }
        if w * 9 < h * 16 { let nh = w * 9 / 16; return CGRect(x: 0, y: (h - nh) / 2, width: w, height: nh) }
        return CGRect(x: 0, y: 0, width: w, height: h)
    }

    /// A box in fractions of the 16:9 crop, as fractions of the whole w x h picture.
    public static func uncrop(_ b: Box, _ w: Int, _ h: Int) -> Box {
        let c = crop16x9(w, h), W = Double(w), H = Double(h)
        return Box(top: (c.minY + b.top * c.height) / H, bottom: (c.minY + b.bottom * c.height) / H,
                   left: (c.minX + b.left * c.width) / W, right: (c.minX + b.right * c.width) / W)
    }

    /// The centre 16:9 crop of `image` (not for portrait pictures, which are letterboxed when drawn).
    public static func cropped(_ image: CGImage) -> CGImage {
        let c = crop16x9(image.width, image.height)
        return c.size == CGSize(width: image.width, height: image.height) ? image : image.cropping(to: c)!
    }

    /// `image` (any shape: it is centre-cropped to 16:9 first, or letterboxed if portrait) as the model's input, channels first.
    public static func pixels(_ image: CGImage, width: Int = 640, height: Int = 352) -> [Float] {
        let fw = width, fh = width * 9 / 16  // the 16:9 frame, 640x360
        let px: [UInt8]
        if image.height > image.width {
            let pw = Double(fh) * Double(image.width) / Double(image.height)
            px = rgba(image, fw, fh, in: CGRect(x: (Double(fw) - pw) / 2, y: 0, width: pw, height: Double(fh)))
        } else {
            px = rgba(cropped(image), fw, fh)
        }
        let rs = resample(px, fw, fh, width, height)
        let n = width * height
        var out = [Float](repeating: 0, count: 3 * n)
        for i in 0..<n {
            for c in 0..<3 { out[c * n + i] = Float(rs[i * 4 + c]) / 127.5 - 1 }
        }
        return out
    }

    /// `image` drawn at w x h, RGBA 8-bit, rows top to bottom. ponytail: CoreGraphics' scaler for the camera frame to
    /// 640x360, as the page's canvas does it (not exact either); a 640x360 picture is copied as it is.
    static func rgba(_ image: CGImage, _ w: Int, _ h: Int, in rect: CGRect? = nil) -> [UInt8] {
        var out = [UInt8](repeating: 0, count: w * h * 4)
        out.withUnsafeMutableBytes { buf in
            let ctx = CGContext(data: buf.baseAddress, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                                space: image.colorSpace ?? CGColorSpace(name: CGColorSpace.sRGB)!,
                                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
            ctx.interpolationQuality = .high
            ctx.draw(image, in: rect ?? CGRect(x: 0, y: 0, width: w, height: h))  // the rest stays black
        }
        return out
    }

    /// PIL's Image.resize(..., BICUBIC) on RGBA 8-bit (libImaging/Resample.c): separable, antialiased when shrinking,
    /// fixed-point weights; the horizontal pass first, each pass only when its size changes. Bit-exact with PIL.
    static func resample(_ src: [UInt8], _ inW: Int, _ inH: Int, _ outW: Int, _ outH: Int) -> [UInt8] {
        var img = src, w = inW
        if outW != inW {
            let (bounds, k, ks) = coefficients(inW, outW)
            var out = [UInt8](repeating: 0, count: outW * inH * 4)
            for y in 0..<inH {
                for x in 0..<outW {
                    let (lo, n) = bounds[x]
                    for c in 0..<4 {
                        var ss = 1 << 21
                        for i in 0..<n { ss += Int(img[(y * w + lo + i) * 4 + c]) * k[x * ks + i] }
                        out[(y * outW + x) * 4 + c] = clip8(ss)
                    }
                }
            }
            img = out; w = outW
        }
        if outH != inH {
            let (bounds, k, ks) = coefficients(inH, outH)
            var out = [UInt8](repeating: 0, count: w * outH * 4)
            for y in 0..<outH {
                let (lo, n) = bounds[y]
                for x in 0..<w {
                    for c in 0..<4 {
                        var ss = 1 << 21
                        for i in 0..<n { ss += Int(img[((lo + i) * w + x) * 4 + c]) * k[y * ks + i] }
                        out[(y * w + x) * 4 + c] = clip8(ss)
                    }
                }
            }
            img = out
        }
        return img
    }

    static func clip8(_ v: Int) -> UInt8 { UInt8(min(255, max(0, v >> 22))) }

    static func bicubic(_ x: Double) -> Double {
        let a = -0.5, x = abs(x)
        if x < 1 { return ((a + 2) * x - (a + 3)) * x * x + 1 }
        if x < 2 { return (((x - 5) * x + 8) * x - 4) * a }
        return 0
    }

    /// Per output pixel, its first input pixel and count, and the fixed-point weights (22 bits), `ks` per pixel.
    static func coefficients(_ inSize: Int, _ outSize: Int) -> ([(Int, Int)], [Int], Int) {
        let scale = Double(inSize) / Double(outSize), filterscale = max(scale, 1), support = 2 * filterscale
        let ks = Int(support.rounded(.up)) * 2 + 1
        var bounds: [(Int, Int)] = [], k = [Int](repeating: 0, count: outSize * ks)
        for xx in 0..<outSize {
            let center = (Double(xx) + 0.5) * scale
            let lo = max(Int(center - support + 0.5), 0), n = min(Int(center + support + 0.5), inSize) - lo
            var w = (0..<n).map { bicubic((Double($0 + lo) - center + 0.5) / filterscale) }
            let ww = w.reduce(0, +)
            if ww != 0 { w = w.map { $0 / ww } }
            for (i, v) in w.enumerated() { k[xx * ks + i] = Int(v < 0 ? -0.5 + v * 4194304 : 0.5 + v * 4194304) }
            bounds.append((lo, n))
        }
        return (bounds, k, ks)
    }
}
