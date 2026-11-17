// d1's answers from the exported function's logits: mlx-vlm's readout (d1_lfm3_vl prompt / box / calibration / grid),
// in Swift. The layout comes from d1.json (coreai/export_app.py), for the questions in prompt/ (the model's inputs).
import Foundation

public struct Layout: Decodable, Sendable {
    public struct Question: Decodable, Sendable {
        public let name: String
        public let type: String  // "choice" or "box"
        public let slots: [Int]
        public let keys: [String]?
    }
    struct Slot: Decodable, Sendable {
        let forms: [[Int]]  // per option, the logits' columns it scores as its best
        let temperature: Double
    }
    struct EdgeGrid: Decodable, Sendable { let low: Double, high: Double, size: Int }

    public let inputTokens: Int
    public let rows: Int     // the function's logits are [rows x columns], padded: the first slots.count rows are ours
    public let columns: Int
    public let prompt: [String: [Int]]  // the prompt inputs' shapes (prompt/<name>.f16)
    public let width: Int
    public let height: Int
    public let questions: [Question]
    let slots: [Slot]
    let edgeGrid: EdgeGrid
    let tables: [String: [[Double]]]

    public static func bundled() throws -> Layout {
        let url = Bundle.module.url(forResource: "d1", withExtension: "json")!
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(Layout.self, from: Data(contentsOf: url))
    }
}

public struct Box: Equatable, Sendable {
    public var top: Double, bottom: Double, left: Double, right: Double
    public init(top: Double, bottom: Double, left: Double, right: Double) {
        self.top = top; self.bottom = bottom; self.left = left; self.right = right
    }
}

public struct ChoiceAnswer: Sendable {
    public let keys: [String]
    public let probabilities: [Double]  // in the order of keys
    public var choice: String { keys[best] }
    public var confidence: Double { probabilities[best] }
    var best: Int { probabilities.indices.max { probabilities[$0] < probabilities[$1] }! }
}

public struct BoxAnswer: Sendable {
    public let box: Box?  // nil when not found
    public let found: Double
    public let confidence: Double
}

public struct Answers: Sendable {
    public var choices: [String: ChoiceAnswer] = [:]
    public var boxes: [String: BoxAnswer] = [:]
}

extension Layout {
    /// The answers for `logits`, row-major [rows x columns].
    public func readout(_ logits: [Float]) -> Answers {
        precondition(logits.count == rows * columns, "expected \(rows) x \(columns) logits")
        let probs = slots.indices.map { k -> [Double] in
            let row = k * columns, s = slots[k]
            return softmax(s.forms.map { $0.map { Double(logits[row + $0]) }.max()! / s.temperature })
        }
        var out = Answers()
        for q in questions {
            let p = q.slots.map { probs[$0] }
            if q.type == "box" {
                out.boxes[q.name] = box(p)
            } else {
                out.choices[q.name] = ChoiceAnswer(keys: q.keys ?? [], probabilities: p[0])
            }
        }
        return out
    }

    /// A box: presence ([yes, no]) then the four edges' distributions (top, bottom, left, right).
    func box(_ p: [[Double]]) -> BoxAnswer {
        let found = through(p[0][0], tables["found:box"] ?? [])
        if found < 0.5 { return BoxAnswer(box: nil, found: found, confidence: 0) }
        let g = Grid(edgeGrid)
        let b = g.rectangle(p[1...].map(g.median))
        return BoxAnswer(box: b, found: found, confidence: through(g.confidence(Array(p[1...]), b), tables["box"] ?? []))
    }
}

func softmax(_ s: [Double]) -> [Double] {
    let m = s.max()!, e = s.map { exp($0 - m) }, t = e.reduce(0, +)
    return e.map { $0 / t }
}

/// `value` through increasing (raw, calibrated) knots, linear between them, flat past the ends.
func through(_ value: Double, _ knots: [[Double]]) -> Double {
    if knots.isEmpty { return value }
    let i = knots.firstIndex { $0[0] >= value } ?? knots.count  // bisect_left
    if i == 0 { return knots[0][1] }
    if i == knots.count { return knots[knots.count - 1][1] }
    let a = knots[i - 1], b = knots[i]
    return a[1] + (value - a[0]) * (b[1] - a[1]) / (b[0] - a[0])
}

/// The first index of a sorted array whose value is >= x.
func bisectLeft(_ a: [Double], _ x: Double) -> Int {
    var lo = 0, hi = a.count
    while lo < hi { let mid = (lo + hi) / 2; if a[mid] < x { lo = mid + 1 } else { hi = mid } }
    return lo
}

struct Grid {
    let low: Double, high: Double, size: Int, step: Double, values: [Double]
    init(_ g: Layout.EdgeGrid) {
        let step = (g.high - g.low) / Double(g.size - 1)
        low = g.low; high = g.high; size = g.size; self.step = step
        values = (0..<(g.size - 1)).map { g.low + Double($0) * step } + [g.high]
    }

    func median(_ p: [Double]) -> Double {
        let total = p.reduce(0, +)
        var mass = 0.0
        for (v, x) in zip(values, p) { mass += x / total; if mass >= 0.5 { return v } }
        return values[values.count - 1]
    }

    /// Two edges of one axis, the second below or right of the first: crossed ones become one bin about their middle.
    func ordered(_ a: Double, _ b: Double) -> (Double, Double) {
        if a < b { return (a, b) }
        let half = step / 2, middle = min(1 - half, max(half, (a + b) / 2))
        return (middle - half, middle + half)
    }

    func rectangle(_ e: [Double]) -> Box {
        let (t, b) = ordered(e[0], e[1]), (l, r) = ordered(e[2], e[3])
        return Box(top: t, bottom: b, left: l, right: r)
    }

    /// The share of boxes drawn from the edge distributions (Halton points through each CDF) with IoU >= 0.5 against `box`.
    func confidence(_ edges: [[Double]], _ box: Box) -> Double {
        let cdfs = edges.map { row -> [Double] in
            let total = row.reduce(0, +)
            var acc = 0.0
            return row.map { acc += $0 / total; return acc }
        }
        let hits = Halton.points.filter { u in
            iou(rectangle((0..<4).map { values[min(size - 1, bisectLeft(cdfs[$0], u[$0]))] }), box) >= 0.5
        }.count
        return Double(hits) / Double(Halton.points.count)
    }
}

func iou(_ a: Box, _ b: Box) -> Double {
    let area = { (x: Box) in (x.bottom - x.top) * (x.right - x.left) }
    let overlap = max(0, min(a.bottom, b.bottom) - max(a.top, b.top)) * max(0, min(a.right, b.right) - max(a.left, b.left))
    return overlap / (area(a) + area(b) - overlap)
}

enum Halton {
    /// 512 points of a Halton sequence in four dimensions (bases 2, 3, 5, 7): the same draws every call.
    static let points: [[Double]] = (1...512).map { i in
        [2, 3, 5, 7].map { base in
            var i = i, value = 0.0, scale = 1.0
            while i > 0 { scale /= Double(base); value += Double(i % base) * scale; i /= base }
            return value
        }
    }
}
