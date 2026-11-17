// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "D1Kit",
    platforms: [.iOS("27.0"), .macOS("27.0")],
    products: [.library(name: "D1Kit", targets: ["D1Kit"])],
    targets: [
        .target(name: "D1Kit", resources: [.copy("d1.json"), .copy("prompt")], swiftSettings: [.swiftLanguageMode(.v5)]),
        .testTarget(name: "D1KitTests", dependencies: ["D1Kit"], exclude: ["Fixtures"], swiftSettings: [.swiftLanguageMode(.v5)]),
    ]
)
