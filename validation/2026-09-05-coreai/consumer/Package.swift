// swift-tools-version: 6.0
import PackageDescription
let package = Package(
    name: "continuation-repro",
    platforms: [.macOS("27.0")],
    dependencies: [.package(name: "coreai-models", path: "../../..")],
    targets: [
        .executableTarget(name: "bundle-gate", dependencies: [.product(name: "CoreAILM", package: "coreai-models")], path: "Sources"),
        .testTarget(name: "ContinuationTests", dependencies: [.product(name: "CoreAILM", package: "coreai-models")], path: "Tests")
    ]
)
