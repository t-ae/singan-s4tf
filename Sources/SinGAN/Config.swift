import Foundation

enum Config {
    // network params
    static let baseChannels = 32
    
    // pyramid params
    static let scaleFactor: Float = 0.75
    static let imageMinSize = 20
    static let imageMaxSize = 250
    
    // training params
    static let trainEpochsPerLayer = 5000
    static let nDisUpdate = 1
    static let alpha: Float = 50
    static let gamma: Float = 0.1
    static let noiseScaleBase: Float = 0.1
    static let noisePadding: NoisePadding = .noise
    
    // test configuration
    static let superResolutionIter = 5
    
    // plot configuration
    static let tensorBoardLogDir = URL(fileURLWithPath: "./logdir")
}

enum NoisePadding {
    case zero, noise
}
