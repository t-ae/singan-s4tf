import Foundation
import TensorFlow
import TensorBoardX

Context.local.randomSeed = (42, 42)

let imageURL = URL(fileURLWithPath: ProcessInfo.processInfo.arguments[1])
//let imageURL = URL(fileURLWithPath: "/Users/araki/Desktop/t-ae/SinGAN/Input/Images/33039_LR.png")
print("Image: \(imageURL)")
let reals = try ImagePyramid.load(file: imageURL)

var modelStack = ModelStack()
let loss = HingeLoss()

let writer = SummaryWriter(logdir: Config.tensorBoardLogDir)
writer.addText(tag: "sizes", text: String(describing: reals.sizes))
func writeImage(tag: String, image: Tensor<Float>, globalStep: Int = 0) {
    var image = image.squeezingShape()
    image = (image + 1) / 2
    image = image.clipped(min: 0, max: 1)
    writer.addImage(tag: tag, image: image, globalStep: globalStep)
}

let noiseOpt = modelStack.createNoiseOpt(sizes: reals.sizes)

func trainSingleScale() {
    Context.local.learningPhase = .training
    let layer = modelStack.trainedLayers
    let tag = "layer\(layer)"
    let real = reals[layer] // [1, H, W, C]
    writeImage(tag: "\(tag)/real", image: real.squeezingShape())
    
    var (gen, disc) = modelStack.createNewModels()
    
    let optG = Adam(for: gen, learningRate: 5e-4, beta1: 0.5, beta2: 0.999)
    let optD = Adam(for: disc, learningRate: 5e-4, beta1: 0.5, beta2: 0.999)
    
    let currentSize = reals.sizes[layer]
    
    let sizes: [Size] // Up to previous size
    let noiseScale: Float
    if layer == 0 {
        sizes = [currentSize]
        noiseScale = 1
    } else {
        sizes = Array(reals.sizes[..<layer])
        
        var rec = modelStack.generate(sizes: Array(reals.sizes[..<layer]), noises: noiseOpt)
        rec = resizeBilinear(images: rec, newSize: reals.sizes[layer])
        let rmse = sqrt(meanSquaredError(predicted: rec, expected: real))
        noiseScale = Config.noiseScaleBase * rmse.scalarized()
    }
        
    var recBase = modelStack.generate(sizes: sizes, noises: noiseOpt)
    recBase = resizeBilinear(images: recBase, newSize: currentSize)
    recBase = modelStack.zeroPad(recBase)
    
    let steps = Config.trainEpochsPerLayer * Config.nDisUpdate
    for step in 0..<steps {
        if step % 10 == 0 {
            print("Epoch: \(step) noiseScale=\(noiseScale)")
        }
        
        var fakeBase = modelStack.generate(sizes: sizes)
        fakeBase = resizeBilinear(images: fakeBase, newSize: currentSize)
        fakeBase = modelStack.zeroPad(fakeBase)
        
        // Train discriminator
        let 𝛁disc = gradient(at: disc)  { disc -> Tensor<Float> in
            let noise = Tensor<Float>(randomNormal: fakeBase.shape) * noiseScale
            let fake = gen(.init(image: fakeBase, noise: noise))
            let fakeScore = disc(fake)
            let realScore = disc(real)

            writer.addScalar(tag: "\(tag)/D.fakeScore", scalar: fakeScore.mean().scalarized(), globalStep: step)
            writer.addScalar(tag: "\(tag)/D.realScore", scalar: realScore.mean().scalarized(), globalStep: step)

            let lossD = loss.lossD(real: realScore, fake: fakeScore)
            writer.addScalar(tag: "\(tag)/D", scalar: lossD.scalarized(), globalStep: step)
            return lossD
        }
        optD.update(&disc, along: 𝛁disc)
        
//        print(disc.head.conv.conv.filter[0, 0, 0, 0])
        
        // Train generator
        if step % Config.nDisUpdate == 0 {
            let 𝛁gen = gradient(at: gen) { gen -> Tensor<Float> in
                let noise = Tensor<Float>(randomNormal: fakeBase.shape) * noiseScale
                let fake = gen(.init(image: fakeBase, noise: noise))
                let score = disc(fake)
                
                let classLoss = loss.lossG(score)
                
                let rec = gen(.init(image: recBase, noise: noiseOpt[layer]))
                let recLoss = meanSquaredError(predicted: rec, expected: real)
                
                if step % (Config.nDisUpdate *  100) == 0 {
                    writeImage(tag: "\(tag)/fake", image: fake, globalStep: step)
                    writeImage(tag: "\(tag)/rec", image: rec, globalStep: step)
                }
                
                let lossG = classLoss + Config.alpha * recLoss
                writer.addScalar(tag: "\(tag)/G.classLoss", scalar: classLoss.scalarized(), globalStep: step)
                writer.addScalar(tag: "\(tag)/G.recLoss", scalar: recLoss.scalarized(), globalStep: step)
                writer.addScalar(tag: "\(tag)/G", scalar: lossG.scalarized(), globalStep: step)
                return lossG
                
            }
            optG.update(&gen, along: 𝛁gen)
        }
        
        if step % (500 * Config.nDisUpdate) == 0 {
            writer.addHistograms(tag: "G\(layer)/", layer: gen, globalStep: step)
            writer.addHistograms(tag: "D\(layer)/", layer: disc, globalStep: step)
        }
    }
    
    // Training End
    modelStack.append(g: gen, d: disc, noiseScale: noiseScale)
    Context.local.learningPhase = .inference
    
    // plot
    do { // random
        let image = modelStack.generate(sizes: Array(reals.sizes[...layer]))
        writeImage(tag: "\(tag)/FinalRandom", image: image, globalStep: 0)
    }
    do { // reconstruct
        let image = modelStack.generate(sizes: Array(reals.sizes[...layer]), noises: noiseOpt)
        writeImage(tag: "\(tag)/FinalRec", image: image, globalStep: 0)
    }
    writer.flush()
}

func train() {
    for i in 0..<reals.images.count {
        print("Start training: layer=\(i)")
        trainSingleScale()
    }
}

func testMultipleScale() {
    print("testMultipleScale")
    Context.local.learningPhase = .inference
    let initialSizes = [Size(width: 25, height: 25),
                        Size(width: 25, height: 50),
                        Size(width: 40, height: 25)]
    for initialSize in initialSizes {
        var sizes = [initialSize]
        for _ in 1..<modelStack.trainedLayers {
            sizes.append(sizes.last!.scaled(factor: 1/Config.scaleFactor))
        }
        
        let image = modelStack.generate(sizes: sizes)
        let label = String(describing: sizes.last!)
        writeImage(tag: "MultiSize/\(label)", image: image.squeezingShape())
    }
}

func testSuperResolution() {
    print("testSuperResolution")
    Context.local.learningPhase = .inference
    var image = reals.images.last!.expandingShape(at: 0) // [1, H, W, C]
    
    for i in 0..<Config.superResolutionIter {
        let newSize = Size(width: image.shape[2], height: image.shape[1])
            .scaled(factor: 1/Config.scaleFactor)
        image = modelStack.superResolution(image: image, size: newSize)
        writeImage(tag: "SuperResolution", image: image.squeezingShape(), globalStep: i)
    }
}

train()
testMultipleScale()
testSuperResolution()
writer.close()
