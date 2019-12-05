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

func generateThroughGenStack(noises: [Tensor<Float>]? = nil) -> Tensor<Float> {
    var image = Tensor<Float>(zeros: [1, reals.sizes[0].height, reals.sizes[0].width, 3])
    
    for (i, gen) in genStack.enumerated() {
        let noise = noises?[i] ?? sampleNoise(image.shape, scale: noiseScales[i])
        image = gen(.init(image: image, noise: noise))
        
        if i+1 < reals.sizes.count {
            let nextSize = reals.sizes[i+1]
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
    
    for i in 0..<Config.trainStepsPerLayer {
        print("step: \(i) noiseScale=\(noiseScale)")
        
        let image = generateThroughGenStack()
        let reconImage = generateThroughGenStack(noises: noiseOpt) // upsampled
        let noise = sampleNoise(image.shape, scale: noiseScale)
        
        // Train generator
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
        
        // Train discriminator
        let (lossD, 𝛁dis) = discCurrent.valueWithGradient { discCurrent -> Tensor<Float> in
            let fake = genCurrent(.init(image: image, noise: noise))
            let fakeScore = discCurrent(fake).mean()
            let realScore = discCurrent(real).mean()
            let loss = pow(fakeScore, 2) + pow(realScore-1, 2)
            return loss
        }
        optD.update(&discCurrent, along: 𝛁dis)
        
        writer.addScalar(tag: "\(tag)/G", scalar: lossG.scalarized(), globalStep: i)
        writer.addScalar(tag: "\(tag)/D", scalar: lossD.scalarized(), globalStep: i)
    }
    
    // plot
    writer.addImage(tag: "\(tag)/real", image: real.squeezingShape())
    do {
        var image = generateThroughGenStack()
        let noise = sampleNoise(image.shape, scale: noiseScale)
        image = genCurrent(.init(image: image, noise: noise))
        writer.addImage(tag: "\(tag)/random", image: image.squeezingShape())
    }
    do {
        var image = generateThroughGenStack(noises: noiseOpt)
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
testSuperResolution()
writer.flush()
