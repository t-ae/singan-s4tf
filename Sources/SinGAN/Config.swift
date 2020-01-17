import Foundation

struct Config: Codable {
    // network params
    let baseChannels: Int
    
    // pyramid params
    let scaleFactor: Float
    let imageMinSize: Int
    let imageMaxSize: Int
    
    // training params
    let trainEpochsPerLayer: Int
    let nDisUpdate: Int
    let alpha: Float
    let gamma: Float
    let noiseScaleBase: Float
    let noisePadding: NoisePadding
    let ganLoss: GANLossType
    let recLoss: ReconstructionLossType
    
    // model params
    let enableSN: GDPair<Bool>
    let enableNorm: GDPair<Bool>
    
    // test configuration
    let superResolutionIter: Int
}

enum NoisePadding: String, Codable {
    case zero, noise
}

struct GDPair<T: Codable>: Codable {
    var G: T
    var D: T
}
