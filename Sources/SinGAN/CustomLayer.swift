import Foundation
import TensorFlow

public func heNormal<Scalar: TensorFlowFloatingPoint>() -> ParameterInitializer<Scalar> {
    return { shape in
        let out = shape.dimensions.dropLast().reduce(1, *)
        return Tensor(randomNormal: shape) * sqrt(2 / Scalar(out))
    }
}

private func l2normalize<Scalar: TensorFlowFloatingPoint>(_ tensor: Tensor<Scalar>) -> Tensor<Scalar> {
    tensor * rsqrt(tensor.squared().sum() + 1e-8)
}

public struct SNConv2D<Scalar: TensorFlowFloatingPoint>: Layer {
    public var conv: Conv2D<Scalar>
    
    @noDerivative
    public var enabled: Bool
    
    @noDerivative
    public let numPowerIterations = 1
    
    @noDerivative
    public let v: Parameter<Scalar>
    
    public init(_ conv: Conv2D<Scalar>, enabled: Bool = true) {
        self.conv = conv
        self.enabled = enabled
        v = Parameter(Tensor(randomNormal: [1, conv.filter.shape[3]]))
    }
    
    @differentiable
    public func wBar() -> Tensor<Scalar> {
        guard enabled else {
            return conv.filter
        }
        let outputDim = conv.filter.shape[3]
        let mat = conv.filter.reshaped(to: [-1, outputDim])
        
        var u = Tensor<Scalar>(0)
        for _ in 0..<numPowerIterations {
            u = l2normalize(matmul(v.value, mat.transposed())) // [1, rows]
            v.value = l2normalize(matmul(u, mat)) // [1, cols]
        }
        
        let sigma = matmul(matmul(u, mat), v.value.transposed()) // [1, 1]
        
        // Should detach sigma?
        return conv.filter / sigma
    }
    
    @differentiable
    public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        // FIXME: tensorflow-RELEASE-0.6 has bug in AD (already fixed in nightly).
        // `f = wBar()` leads 0 gradients.
        let f: Tensor<Scalar>
        if enabled {
            f = conv.filter
        } else {
            let outputDim = conv.filter.shape[3]
            let mat = conv.filter.reshaped(to: [-1, outputDim])

            var u = Tensor<Scalar>(0)
            for _ in 0..<numPowerIterations {
                u = l2normalize(matmul(v.value, mat.transposed())) // [1, rows]
                v.value = l2normalize(matmul(u, mat)) // [1, cols]
            }

            let sigma = matmul(matmul(u, mat), v.value.transposed()) // [1, 1]

            // Should detach sigma?
            f = conv.filter / sigma
        }
        return conv.activation(conv2D(
            input,
            filter: f,
            strides: (1, conv.strides.0, conv.strides.1, 1),
            padding: conv.padding,
            dilations: (1, conv.dilations.0, conv.dilations.1, 1)) + conv.bias)
    }
}

public struct InstanceNorm2D<Scalar: TensorFlowFloatingPoint>: Layer {
    var scale: Tensor<Scalar>
    var offset: Tensor<Scalar>
    
    public init(featureCount: Int) {
        scale = Tensor(ones: [featureCount])
        offset = Tensor(zeros: [featureCount])
    }
    
    @differentiable
    public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        precondition(input.rank == 4)
        
        let mean = input.mean(alongAxes: 1, 2)
        let variance = squaredDifference(input, mean).mean(alongAxes: 1, 2)
        let normalized = (input - mean) * rsqrt(variance + 1e-8)
        
        return scale * normalized + offset
    }
}
