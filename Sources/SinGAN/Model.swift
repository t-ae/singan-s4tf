import TensorFlow

func resizeBilinear(images: Tensor<Float>, newSize: Size) -> Tensor<Float> {
    return _Raw.resizeBilinear(images: images,
                               size: Tensor([Int32(newSize.height), Int32(newSize.width)]),
                               alignCorners: true)
}

struct ConvBlock: Layer {
    typealias Activation = @differentiable (Tensor<Float>) -> Tensor<Float>
    
    var conv: Conv2D<Float>
    var bn: BatchNorm<Float>
    @noDerivative public let activation: Activation
    
    init(inputDim: Int, outputDim: Int) {
        self.conv = Conv2D(filterShape: (Config.kernelSize, Config.kernelSize, inputDim, outputDim),
                           padding: .same)
        self.bn = BatchNorm(featureCount: outputDim)
        self.activation = { leakyRelu($0) }
    }
    
    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        activation(bn(conv(input)))
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
    var tail: Conv2D<Float>
    
    init() {
        self.head = ConvBlock(inputDim: 3, outputDim: 32)
        self.conv1 = ConvBlock(inputDim: 32, outputDim: 32)
        self.conv2 = ConvBlock(inputDim: 32, outputDim: 32)
        self.conv3 = ConvBlock(inputDim: 32, outputDim: 32)
        self.tail = Conv2D(filterShape: (3, 3, 32, 3), padding: .same)
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
        return x + input.image
    }
}

struct Discriminator: Layer {
    var head: ConvBlock
    var conv1: ConvBlock
    var conv2: ConvBlock
    var conv3: ConvBlock
    var tail: Conv2D<Float>
    
    init() {
        self.head = ConvBlock(inputDim: 3, outputDim: 32)
        self.conv1 = ConvBlock(inputDim: 32, outputDim: 32)
        self.conv2 = ConvBlock(inputDim: 32, outputDim: 32)
        self.conv3 = ConvBlock(inputDim: 32, outputDim: 32)
        self.tail = Conv2D(filterShape: (3, 3, 32, 1), padding: .same)
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
