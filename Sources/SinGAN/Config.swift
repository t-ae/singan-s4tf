import Foundation

enum Config {
    // network params
    static let kernelSize = 3
    static let baseChannels = 32
    
    // pyramid params
    static let scaleFactor: Float = 0.75
    static let noiseAmplitude: Float = 0.1
    static let imageMinSize = 25
    static let imageMaxSize = 250
    
    // training params
    static let trainStepsPerLayer = 3000
    static let alpha: Float = 50
    
    // test configuration
    static let superResolutionIter = 8
    
    // plot configuration
    static let tensorBoardLogDir = URL(fileURLWithPath: "/tmp/SinGAN")
}
