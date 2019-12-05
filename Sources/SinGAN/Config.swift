import Foundation

enum Config {
    // network params
    static let kernelSize = 3
    static let intermediateChannels = 32
    
    // pyramid params
    static let scaleFactor: Float = 0.75
    static let noiseAmplitude: Float = 0.1
    static let imageMinSize = 25
    static let imageMaxSize = 250
    
    // training params
    static let trainStepsPerLayer = 3000
    static let alpha: Float = 50
    
    // test configuration
    static let superResolutionTimes = 4
}
