import TensorFlow

@differentiable
func lrelu(_ x: Tensor<Float>) -> Tensor<Float> {
    leakyRelu(x, alpha: 0.2)
}

func resizeBilinear(images: Tensor<Float>, newSize: Size) -> Tensor<Float> {
    _Raw.resizeBilinear(images: images,
                        size: Tensor([Int32(newSize.height), Int32(newSize.width)]),
                        alignCorners: true)
}
