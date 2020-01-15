import TensorFlow
import TensorBoardX

private let kernelSize = 3

@differentiable
func lrelu(_ x: Tensor<Float>) -> Tensor<Float> {
    leakyRelu(x)
}

@differentiable
func hingeLossD(real: Tensor<Float>, fake: Tensor<Float>) -> Tensor<Float> {
    relu(1 - real).mean() + relu(1+fake).mean()
}

@differentiable
func hingeLossG(_ x: Tensor<Float>) -> Tensor<Float> {
    -x
}

func resizeBilinear(images: Tensor<Float>, newSize: Size) -> Tensor<Float> {
    _Raw.resizeBilinear(images: images,
                        size: Tensor([Int32(newSize.height), Int32(newSize.width)]),
                        alignCorners: true)
}

public func convWeightInit<Scalar: TensorFlowFloatingPoint>(
    seed: TensorFlowSeed = Context.local.randomSeed
) -> ParameterInitializer<Scalar> {
    { Tensor<Scalar>(randomNormal: $0, mean: Tensor(0), standardDeviation: Tensor(0.02), seed: seed) }
}

struct ConvBlock: Layer {
    var conv: SNConv2D<Float>
    var bn: BatchNorm<Float>
    
    @noDerivative
    let enableBatchNorm: Bool
    
    init(inputChannels: Int,
         outputChannels: Int,
         enableSpectralNorm: Bool = true,
         enableBatchNorm: Bool = true) {
        self.conv = SNConv2D(Conv2D(filterShape: (3, 3, inputChannels, outputChannels),
                                    filterInitializer: convWeightInit()),
                             enabled: enableSpectralNorm)
        self.bn = BatchNorm(featureCount: outputChannels)
        self.enableBatchNorm = enableBatchNorm
    }
    
    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        var x = input
        x = conv(x)
        if enableBatchNorm {
            x = bn(x)
        }
        x = lrelu(x)
        return x
    }
}

struct Generator: Layer {
    struct Input: Differentiable {
        var image: Tensor<Float> // [batch_size, height, width, 3]
        var noise: Tensor<Float> // [batch_size, height, width, 3]
    }

    var head: ConvBlock
    var conv1: ConvBlock
    var conv2: ConvBlock
    var conv3: ConvBlock
    var tail: SNConv2D<Float>
    
    init(channels: Int) {
        let enableSN = false
        self.head = ConvBlock(inputChannels: 3, outputChannels: channels,
                              enableSpectralNorm: enableSN)
        self.conv1 = ConvBlock(inputChannels: channels, outputChannels: channels,
                               enableSpectralNorm: enableSN)
        self.conv2 = ConvBlock(inputChannels: channels, outputChannels: channels,
                               enableSpectralNorm: enableSN)
        self.conv3 = ConvBlock(inputChannels: channels, outputChannels: channels,
                               enableSpectralNorm: enableSN)
        self.tail = SNConv2D(Conv2D(filterShape: (3, 3, channels, 3),
                                    filterInitializer: convWeightInit()),
                             enabled: enableSN)
    }

    @differentiable
    func callAsFunction(_ input: Input) -> Tensor<Float> {
        var x = input.image + input.noise
        x = head(x)
        x = conv1(x)
        x = conv2(x)
        x = conv3(x)
        x = tail(x)
        x = tanh(x)

        let h = input.image.shape[1]
        let w = input.image.shape[2]
        let unpad = input.image.slice(lowerBounds: [0, 5, 5, 0], upperBounds: [1, h-5, w-5, 3])

        return x + unpad
    }
}

struct Discriminator: Layer {
    var head: ConvBlock
    var conv1: ConvBlock
    var conv2: ConvBlock
    var conv3: ConvBlock
    var tail: SNConv2D<Float>

    init(channels: Int) {
        let enableBN = false
        self.head = ConvBlock(inputChannels: 3, outputChannels: channels,
                              enableBatchNorm: enableBN)
        self.conv1 = ConvBlock(inputChannels: channels, outputChannels: channels,
                               enableBatchNorm: enableBN)
        self.conv2 = ConvBlock(inputChannels: channels, outputChannels: channels,
                               enableBatchNorm: enableBN)
        self.conv3 = ConvBlock(inputChannels: channels, outputChannels: channels,
                               enableBatchNorm: enableBN)
        self.tail = SNConv2D(Conv2D(filterShape: (3, 3, channels, 1),
                                    filterInitializer: convWeightInit()))
    }

    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        var x = input
        x = head(x)
        x = conv1(x)
        x = conv2(x)
        x = conv3(x)
        x = tail(x)
        return x
    }
}
