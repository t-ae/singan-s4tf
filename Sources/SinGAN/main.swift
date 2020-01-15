import Foundation
import TensorFlow
import TensorBoardX

let imageURL = URL(fileURLWithPath: ProcessInfo.processInfo.arguments[1])
//let imageURL = URL(fileURLWithPath: "/Users/araki/Desktop/t-ae/SinGAN/Input/Images/33039_LR.png")
print("Image: \(imageURL)")
let reals = try ImagePyramid.load(file: imageURL)

var genStack: [Generator] = []
var discStack: [Discriminator] = []

let zeropad = ZeroPadding2D<Float>(padding: (5, 5))

let writer = SummaryWriter(logdir: Config.tensorBoardLogDir)
writer.addText(tag: "sizes", text: String(describing: reals.sizes))
func writeImage(tag: String, image: Tensor<Float>, globalStep: Int = 0) {
    var image = image.squeezingShape()
    image = (image + 1) / 2
    writer.addImage(tag: tag, image: image, globalStep: globalStep)
}

func sampleNoise(_ shape: TensorShape, scale: Float = 1.0) -> Tensor<Float> {
    Tensor<Float>(randomNormal: shape)
}

let noiseOpt = [zeropad(sampleNoise(reals[0].shape))]
    + reals.images[1...].map { zeropad(Tensor<Float>(zeros: $0.shape)) }
var noiseScales = [Float]()

func generateThroughGenStack(
    sizes: [Size],
    noises: [Tensor<Float>]? = nil // pass fixed noises for reconstrution
) -> Tensor<Float> {
    var image = Tensor<Float>(zeros: [1, sizes[0].height, sizes[0].width, 3])
    
    for (i, gen) in genStack.enumerated() {
        let img = zeropad(image)
        let noise = noises?[i] ?? sampleNoise(img.shape, scale: noiseScales[i])
        image = gen(.init(image: img, noise: noise))
        
        if i+1 < sizes.count {
            let nextSize = sizes[i+1]
            image = resizeBilinear(images: image, newSize: nextSize)
        }
    }
    
    return image
}

func trainSingleScale() {
    Context.local.learningPhase = .training
    let layer = genStack.count
    let tag = "layer\(layer)"
    let real = reals[layer] // [1, H, W, C]
    writeImage(tag: "\(tag)/real", image: real.squeezingShape())
    
    var genCurrent: Generator
    var discCurrent: Discriminator
    
    if layer == 0 {
        genCurrent = Generator(channels: Config.baseChannels)
        discCurrent = Discriminator(channels: Config.baseChannels)
    } else {
        // Copy from previous
        genCurrent = genStack.last!
        discCurrent = discStack.last!
    }
    
    let optG = Adam(for: genCurrent, learningRate: 5e-4, beta1: 0.5, beta2: 0.999)
    let optD = Adam(for: discCurrent, learningRate: 5e-4, beta1: 0.5, beta2: 0.999)
    
    let noiseScale: Float
    if layer == 0 {
        noiseScale = 1
    } else {
        let recon = generateThroughGenStack(sizes: reals.sizes, noises: noiseOpt)
        let rmse = sqrt((pow((recon - real), 2)).mean())
        noiseScale = Config.noiseScaleBase * rmse.scalarized()
    }
    
    let steps = Config.trainEpochsPerLayer * Config.nDisUpdate
    
    for step in 0..<steps {
        if step % 10 == 0 {
            print("Epoch: \(step) noiseScale=\(noiseScale)")
        }
        
        let reconImage = zeropad(generateThroughGenStack(sizes: reals.sizes, noises: noiseOpt))
        let image = zeropad(generateThroughGenStack(sizes: reals.sizes))
        
        // Train discriminator
        let 𝛁dis = gradient(at: discCurrent)  { discCurrent -> Tensor<Float> in
            let noise = sampleNoise(image.shape, scale: noiseScale)
            let fake = genCurrent(.init(image: image, noise: noise))
            let fakeScore = discCurrent(fake).mean()
            let realScore = discCurrent(real).mean()
            
            writer.addScalar(tag: "\(tag)/Dfake", scalar: fakeScore.scalarized(), globalStep: step)
            writer.addScalar(tag: "\(tag)/Dreal", scalar: realScore.scalarized(), globalStep: step)
            
            let lossD = hingeLossD(real: realScore, fake: fakeScore)
            writer.addScalar(tag: "\(tag)/D", scalar: lossD.scalarized(), globalStep: step)
            return lossD
        }
        optD.update(&discCurrent, along: 𝛁dis)
        
        // Train generator
        if step % Config.nDisUpdate == 0 {
            let 𝛁gen = gradient(at: genCurrent) { genCurrent -> Tensor<Float> in
                let noise = sampleNoise(image.shape, scale: noiseScale)
                let fake = genCurrent(.init(image: image, noise: noise))
                let score = discCurrent(fake).mean()
                
                let classLoss = hingeLossG(score)
                
                let recon = genCurrent(.init(image: reconImage, noise: noiseOpt[layer]))
                let reconLoss = meanSquaredError(predicted: recon, expected: real)
                
                if step % (Config.nDisUpdate *  100) == 0 {
                    writeImage(tag: "\(tag)/fake_image", image: fake, globalStep: step)
                    writeImage(tag: "\(tag)/recon", image: recon, globalStep: step)
                }
                
                let lossG = classLoss + Config.alpha * reconLoss
                writer.addScalar(tag: "\(tag)/Gclass", scalar: classLoss.scalarized(), globalStep: step)
                writer.addScalar(tag: "\(tag)/Grec", scalar: reconLoss.scalarized(), globalStep: step)
                writer.addScalar(tag: "\(tag)/G", scalar: lossG.scalarized(), globalStep: step)
                return lossG
                
            }
            optG.update(&genCurrent, along: 𝛁gen)
            
        }
        
        if step % (100 * Config.nDisUpdate) == 0 {
//            genCurrent.writeHistogram(writer: writer, layer: layer, globalStep: step)
//            discCurrent.writeHistogram(writer: writer, layer: layer, globalStep: step)
        }
    }
    
    Context.local.learningPhase = .inference
    
    // plot
    do {
        var image = zeropad(generateThroughGenStack(sizes: reals.sizes))
        let noise = sampleNoise(image.shape, scale: noiseScale)
        image = genCurrent(.init(image: image, noise: noise))
        writeImage(tag: "\(tag)/random", image: image, globalStep: 0)
    }
    do {
        var image = zeropad(generateThroughGenStack(sizes: reals.sizes, noises: noiseOpt))
        let noise = noiseOpt[layer]
        image = genCurrent(.init(image: image, noise: noise))
        writeImage(tag: "\(tag)/reconstruct", image: image, globalStep: 0)
    }
    writer.flush()
    
    // end training
    genStack.append(genCurrent)
    discStack.append(discCurrent)
    noiseScales.append(noiseScale)
    print(noiseScales)
}

func train() {
    for i in 0..<reals.images.count {
        print("Start training: layer=\(i)")
        trainSingleScale()
    }
}

func testMultipleScale() {
    Context.local.learningPhase = .inference
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
        writeImage(tag: "MultiSize/\(label)", image: image.squeezingShape())
    }
}

func testSuperResolution() {
    Context.local.learningPhase = .inference
    var image = reals.images.last!.expandingShape(at: 0) // [1, H, W, C]
    let gen = genStack.last!
    
    for i in 0..<Config.superResolutionIter {
        let newSize = Size(width: image.shape[2], height: image.shape[1])
            .scaled(factor: 1/Config.scaleFactor)
        image = resizeBilinear(images: image, newSize: newSize)
        
        let noise = sampleNoise(image.shape, scale: noiseScales.last!)
        image = gen(.init(image: zeropad(image), noise: zeropad(noise)))
        
        writeImage(tag: "SuperResolution", image: image.squeezingShape(), globalStep: i)
    }
}

train()
testMultipleScale()
testSuperResolution()
writer.close()
