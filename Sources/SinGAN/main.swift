import Foundation
import TensorFlow
import TensorBoardX

Context.local.randomSeed = (42, 42)

let imageURL = URL(fileURLWithPath: ProcessInfo.processInfo.arguments[1])
print("Image: \(imageURL)")

let config = Config(
    baseChannels: 32,
    scaleFactor: 0.75,
    imageMinSize: 20,
    imageMaxSize: 250,
    trainEpochsPerLayer: 5000,
    nDisUpdate: 1,
    alpha: 20,
    gamma: 0.1,
    noiseScaleBase: 0.1,
    noisePadding: .zero,
    ganLoss: .hinge,
    recLoss: .meanSquaredError,
    enableSN: .init(G: false, D: true),
    enableNorm: .init(G: true, D: false),
    superResolutionIter: 5,
    tensorBoardLogDir: URL(fileURLWithPath: "./logdir/\(imageURL.deletingPathExtension().lastPathComponent)")
)


let reals = try ImagePyramid.load(file: imageURL, config: config)

var modelStack = ModelStack(config: config)
let ganLossCriterion = GANLoss(type: config.ganLoss)
let recLossCriterion = ReconstructionLoss(type: config.recLoss)

let writer = SummaryWriter(logdir: config.tensorBoardLogDir)
writer.addText(tag: "sizes", text: String(describing: reals.sizes))
try writer.addJSONText(tag: "config", encodable: config)
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
        noiseScale = config.noiseScaleBase * rmse.scalarized()
    }
    
    var recBase = modelStack.generate(sizes: sizes, noises: noiseOpt)
    recBase = resizeBilinear(images: recBase, newSize: currentSize)
    recBase = modelStack.zeroPad(recBase)
    
    let steps = config.trainEpochsPerLayer * config.nDisUpdate
    for step in 0..<steps {
        if step % 10 == 0 {
            print("Epoch: \(step) noiseScale=\(noiseScale)")
        }
        if step == steps * 8 / 10 {
            print("Decay leraning rate")
            optG.learningRate *= config.gamma
            optD.learningRate *= config.gamma
        }
        
        var fakeBase = modelStack.generate(sizes: sizes)
        fakeBase = resizeBilinear(images: fakeBase, newSize: currentSize)
        fakeBase = modelStack.zeroPad(fakeBase)
        
        // Train discriminator
        let 𝛁disc = gradient(at: disc)  { disc -> Tensor<Float> in
            let noise = modelStack.sampleNoise(for: currentSize, noiseScale: noiseScale)
            let fake = gen(.init(image: fakeBase, noise: noise))
            let fakeScore = disc(fake)
            let realScore = disc(real)

            writer.addScalar(tag: "\(tag)D/fakeScoreMean",
                scalar: fakeScore.mean().scalarized(),
                             globalStep: step)
            writer.addScalar(tag: "\(tag)D/realScoreMean",
                scalar: realScore.mean().scalarized(),
                             globalStep: step)

            let lossD = ganLossCriterion.lossD(real: realScore, fake: fakeScore)
            writer.addScalar(tag: "\(tag)D/loss", scalar: lossD.scalarized(), globalStep: step)
            return lossD
        }
        optD.update(&disc, along: 𝛁disc)
        
        // Train generator
        if step % config.nDisUpdate == 0 {
            let 𝛁gen = gradient(at: gen) { gen -> Tensor<Float> in
                let noise = modelStack.sampleNoise(for: currentSize, noiseScale: noiseScale)
                let fake = gen(.init(image: fakeBase, noise: noise))
                let score = disc(fake)
                
                let classLoss = ganLossCriterion.lossG(score)
                
                let rec = gen(.init(image: recBase, noise: noiseOpt[layer]))
                let recLoss = recLossCriterion(real: real, fake: rec)
                
                if step % (config.nDisUpdate *  100) == 0 {
                    writeImage(tag: "\(tag)/fake", image: fake, globalStep: step)
                    writeImage(tag: "\(tag)/rec", image: rec, globalStep: step)
                }
                
                let lossG = classLoss + config.alpha * recLoss
                writer.addScalar(tag: "\(tag)G/classLoss",
                    scalar: classLoss.scalarized(),
                                 globalStep: step)
                writer.addScalar(tag: "\(tag)G/recLoss", scalar: recLoss.scalarized(), globalStep: step)
                writer.addScalar(tag: "\(tag)G/loss", scalar: lossG.scalarized(), globalStep: step)
                return lossG
                
            }
            optG.update(&gen, along: 𝛁gen)
        }
        
        if step % (500 * config.nDisUpdate) == 0 {
            writer.addHistograms(tag: "\(tag)G/", layer: gen, globalStep: step)
            writer.addHistograms(tag: "\(tag)D/", layer: disc, globalStep: step)
        }
    }
    
    // Training End
    modelStack.append(g: gen, d: disc, noiseScale: noiseScale)
    Context.local.learningPhase = .inference
    
    // plot
    for i in 0..<10 { // random
        let image = modelStack.generate(sizes: Array(reals.sizes[...layer]))
        writeImage(tag: "\(tag)/FinalRandom", image: image, globalStep: i)
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
                        Size(width: 40, height: 25),
                        Size(width: 80, height: 25)]
    for initialSize in initialSizes {
        var sizes = [initialSize]
        for _ in 1..<modelStack.trainedLayers {
            sizes.append(sizes.last!.scaled(factor: 1/config.scaleFactor))
        }
        
        let image = modelStack.generate(sizes: sizes)
        let label = String(describing: sizes.last!)
        writeImage(tag: "MultiSize/\(label)", image: image.squeezingShape())
    }
}

func testSuperResolution() {
    print("testSuperResolution")
    Context.local.learningPhase = .inference
    var image = reals.images.last! // [1, H, W, C]
    
    for i in 0..<config.superResolutionIter {
        let newSize = Size(width: image.shape[2], height: image.shape[1])
            .scaled(factor: 1/config.scaleFactor)
        image = modelStack.superResolution(image: image, size: newSize)
        writeImage(tag: "SuperResolution", image: image.squeezingShape(), globalStep: i)
    }
}

train()
testMultipleScale()
testSuperResolution()
writer.close()
