// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "d1worker",
    platforms: [.macOS("27.0")],
    targets: [.executableTarget(name: "d1worker", swiftSettings: [.swiftLanguageMode(.v5)])]
)
