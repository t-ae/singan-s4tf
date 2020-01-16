import Foundation
import TensorFlow

struct ModelStack {
    private(set) var generators: [Generator]
    private(set) var discriminators: [Discriminator]
    private(set) var noiseScales: [Float]
    
    var trainedLayers: Int {
        generators.count
    }
    
    init() {
        generators = []
        discriminators = []
        noiseScales = []
    }
    
    func createNewModels() -> (Generator, Discriminator) {
        if let g = generators.last, let d = discriminators.last {
            // Copy of previous models
            return (g, d)
        } else {
            let generator = Generator(channels: Config.baseChannels)
            let discriminator = Discriminator(channels: Config.baseChannels)
            return (generator, discriminator)
        }
    }
    
    mutating func append(g: Generator, d: Discriminator, noiseScale: Float) {
        generators.append(g)
        discriminators.append(d)
        noiseScales.append(noiseScale)
    }
    
    let zeroPad = ZeroPadding2D<Float>(padding: (5, 5))
    
    func createNoiseOpt(sizes: [Size]) -> [Tensor<Float>] {
        var noises = [Tensor<Float>]()
        
        noises.append(Tensor<Float>(randomNormal: noiseShape(for: sizes[0])))
        for size in sizes.dropFirst() {
            noises.append(Tensor<Float>(zeros: noiseShape(for: size)))
        }
        
        return noises
    }
    
    func noiseShape(for size: Size) -> TensorShape {
        // padding 5
        return [1, size.height + 5*2, size.width + 5*2, 3]
    }
    
    func generate(sizes: [Size]) -> Tensor<Float> {
        let noises = zip(sizes, noiseScales).map { size, noiseScale in
            Tensor<Float>(randomNormal: noiseShape(for: size)) * noiseScale
        }
        return generate(sizes: sizes, noises: noises)
    }
    
    func generate(sizes: [Size], noises: [Tensor<Float>]) -> Tensor<Float> {
        var image = Tensor<Float>(zeros: [1, sizes[0].height, sizes[0].width, 3])
        if generators.isEmpty {
            return image
        }
        
        precondition(sizes.count == generators.count)
        
        image = generate(image: image, index: 0, noise: noises[0])
        
        for index in 1..<sizes.count {
            image = resizeBilinear(images: image, newSize: sizes[index])
            image = generate(image: image, index: index, noise: noises[index])
        }
        
        return image
    }
    
    func generate(image: Tensor<Float>, index: Int, noise: Tensor<Float>) -> Tensor<Float> {
        let gen = generators[index]
        let image = zeroPad(image)
        return gen(.init(image: image, noise: noise))
    }
    
    func superResolution(image: Tensor<Float>, size: Size) -> Tensor<Float> {
        let gen = generators.last!
        let noiseScale = noiseScales.last!
        
        var image = image
        image = resizeBilinear(images: image, newSize: size)
        image = zeroPad(image)
        let noise = Tensor<Float>(randomNormal: noiseShape(for: size)) * noiseScale
        
        return gen(.init(image: image, noise: noise))
    }
}