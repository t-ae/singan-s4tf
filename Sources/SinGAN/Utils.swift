import TensorFlow
import GANUtils

@differentiable
func lrelu(_ x: Tensor<Float>) -> Tensor<Float> {
    leakyRelu(x, alpha: 0.2)
}

func resizeBilinear(images: Tensor<Float>, newSize: Size) -> Tensor<Float> {
    resizeBilinear(images: images, width: newSize.width, height: newSize.height, alignCorners: true)
}

public func heNormal<Scalar: TensorFlowFloatingPoint>() -> ParameterInitializer<Scalar> {
    return { shape in
        let out = shape.dimensions.dropLast().reduce(1, *)
        return Tensor(randomNormal: shape) * sqrt(2 / Scalar(out))
    }
}
