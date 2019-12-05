import Foundation
import TensorFlow
import TensorBoardX

let imageURL = URL(fileURLWithPath: "/Users/araki/Desktop/t-ae/SinGAN/Input/Images/balloons.png")
let reals = try ImagePyramid.load(file: imageURL)

var genStack: [Generator] = []

let writer = SummaryWriter(logdir: Config.tensorBoardLogDir)
writer.addText(tag: "sizes", text: String(describing: reals.sizes))

func sampleNoise(_ shape: TensorShape, scale: Float = 1.0) -> Tensor<Float> {
    Tensor<Float>(randomNormal: shape)
}

let noiseOpt = [sampleNoise(reals[0].shape)] + reals.images[1...].map { Tensor<Float>(zeros: $0.shape) }
var noiseScales = noiseOpt.map { _ in 1.0 as Float }

func generateThroughGenStack(
    sizes: [Size],
    noises: [Tensor<Float>]? = nil // pass fixed noises for reconstrution
) -> Tensor<Float> {
    var image = Tensor<Float>(zeros: [1, sizes[0].height, sizes[0].width, 3])
    
    for (i, gen) in genStack.enumerated() {
        let noise = noises?[i] ?? sampleNoise(image.shape, scale: noiseScales[i])
        image = gen(.init(image: image, noise: noise))
        
        if i+1 < sizes.count {
            let nextSize = sizes[i+1]
            image = resizeBilinear(images: image, newSize: nextSize)
        }
    }
    
    return image
}

func trainSingleScale() {
    let layer = genStack.count
    let tag = "layer\(layer)"
    let real = reals[layer].expandingShape(at: 0)
    
    // increase this number by a factor of 2 every 4 scales
    let numChannels = Config.baseChannels * (1 << (layer/4))
    
    var genCurrent = Generator(channels: numChannels)
    var discCurrent = Discriminator(channels: numChannels)
    
    let optG = Adam(for: genCurrent, learningRate: 5e-4, beta1: 0.5, beta2: 0.999)
    let optD = Adam(for: discCurrent, learningRate: 5e-4, beta1: 0.5, beta2: 0.999)
    
    var noiseScale: Float = 1.0
    
    for i in 0..<Config.trainEpochsPerLayer {
        print("Epoch: \(i) noiseScale=\(noiseScale)")
        
        // Inputs
        let image = generateThroughGenStack(sizes: reals.sizes)
        let reconImage = generateThroughGenStack(sizes: reals.sizes, noises: noiseOpt)
        
        // Train discriminator
        var lossDs: Float = 0
        for _ in 0..<Config.discIter {
            let noise = sampleNoise(image.shape, scale: noiseScale)
            
            let (lossD, 𝛁dis) = discCurrent.valueWithGradient { discCurrent -> Tensor<Float> in
                let fake = genCurrent(.init(image: image, noise: noise))
                let fakeScore = discCurrent(fake).mean()
                let realScore = discCurrent(real).mean()
                let loss = pow(fakeScore, 2) + pow(realScore-1, 2)
                return loss
            }
            optD.update(&discCurrent, along: 𝛁dis)
            lossDs += lossD.scalarized()
        }
        
        // Train generator
        var lossGs: Float = 0
        for _ in 0..<Config.genIter {
            let noise = sampleNoise(image.shape, scale: noiseScale)
            
            let (lossG, 𝛁gen) = genCurrent.valueWithGradient { genCurrent -> Tensor<Float> in
                let fake = genCurrent(.init(image: image, noise: noise))
                let score = discCurrent(fake).mean()
                let loss = pow(score - 1, 2)
                let recon = genCurrent(.init(image: reconImage, noise: noiseOpt[layer]))
                let reconLoss = meanSquaredError(predicted: recon, expected: real)
                
                // Update noise scale
                if layer > 0 {
                    noiseScale = sqrt(reconLoss.scalarized())
                }
                
                return loss + Config.alpha * reconLoss
            }
            optG.update(&genCurrent, along: 𝛁gen)
            lossGs += lossG.scalarized()
        }
        
        writer.addScalar(tag: "\(tag)/G", scalar: lossGs / Float(Config.genIter), globalStep: i)
        writer.addScalar(tag: "\(tag)/D", scalar: lossDs / Float(Config.discIter), globalStep: i)
    }
    
    // plot
    writer.addImage(tag: "\(tag)/real", image: real.squeezingShape())
    do {
        var image = generateThroughGenStack(sizes: reals.sizes)
        let noise = sampleNoise(image.shape, scale: noiseScale)
        image = genCurrent(.init(image: image, noise: noise))
        writer.addImage(tag: "\(tag)/random", image: image.squeezingShape())
    }
    do {
        var image = generateThroughGenStack(sizes: reals.sizes, noises: noiseOpt)
        let noise = noiseOpt[layer]
        image = genCurrent(.init(image: image, noise: noise))
        writer.addImage(tag: "\(tag)/reconstruct", image: image.squeezingShape())
    }
    writer.flush()
    
    // end training
    genStack.append(genCurrent)
    
    noiseScales[layer] = noiseScale
    
    print(noiseScales)
}

func train() {
    for i in 0..<reals.images.count {
        print("Start training: layer=\(i)")
        trainSingleScale()
    }
}

func testMultipleScale() {
    let initialSizes = [Size(width: 25, height: 25),
                        Size(width: 25, height: 50),
                        Size(width: 40, height: 25)]
    for initialSize in initialSizes {
        var sizes = [initialSize]
        for _ in 1..<genStack.count {
            sizes.append(sizes.last!.scaled(factor: 1/Config.scaleFactor))
        }
        
        let image = generateThroughGenStack(sizes: sizes)
        let label = String(describing: sizes.last!)
        writer.addImage(tag: "MultiSize/\(label)", image: image.squeezingShape())
    }
}

func testSuperResolution() {
    var image = reals.images.last!
    let gen = genStack.last!
    
    for i in 0..<Config.superResolutionIter {
        let newSize = Size(width: Int(Float(image.shape[2]) / Config.scaleFactor),
                           height: Int(Float(image.shape[1]) / Config.scaleFactor))
        image = resizeBilinear(images: image, newSize: newSize)
        
        let noise = sampleNoise(image.shape, scale: noiseScales.last!)
        image = gen(.init(image: image, noise: noise))
        
        writer.addImage(tag: "SuperResolution", image: image.squeezingShape(), globalStep: i)
    }
}

train()
testMultipleScale()
testSuperResolution()
writer.flush()
