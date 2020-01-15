import Foundation
import TensorFlow

private func l2normalize<Scalar: TensorFlowFloatingPoint>(_ tensor: Tensor<Scalar>) -> Tensor<Scalar> {
    tensor * rsqrt(tensor.squared().sum() + 1e-8)
}

public struct SNDense<Scalar: TensorFlowFloatingPoint>: Layer {
    public var dense: Dense<Scalar>
    
    @noDerivative
    public var enabled: Bool
    
    @noDerivative
    public let numPowerIterations = 1
    
    @noDerivative
    public let v: Parameter<Scalar>
    
    public init(_ dense: Dense<Scalar>, enabled: Bool = true) {
        self.dense = dense
        precondition(dense.weight.rank == 2, "batched dense is not supported.")
        
        self.enabled = enabled
        v = Parameter(Tensor(randomNormal: [1, dense.weight.shape[1]]))
    }
    
    @differentiable
    public func wBar() -> Tensor<Scalar> {
        guard enabled else {
            return dense.weight
        }
        let outputDim = dense.weight.shape[1]
        let mat = dense.weight.reshaped(to: [-1, outputDim])
        
        var u = Tensor<Scalar>(0)
        for _ in 0..<numPowerIterations {
            u = l2normalize(matmul(v.value, mat.transposed())) // [1, rows]
            v.value = l2normalize(matmul(u, mat)) // [1, cols]
        }
        
        let sigma = matmul(matmul(u, mat), v.value.transposed()) // [1, 1]
        
        // Should detach sigma?
        return dense.weight / sigma
    }
    
    @differentiable
    public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        dense.activation(matmul(input, wBar()) + dense.bias)
    }
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
        conv.activation(conv2D(
            input,
            filter: wBar(),
            strides: (1, conv.strides.0, conv.strides.1, 1),
            padding: conv.padding,
            dilations: (1, conv.dilations.0, conv.dilations.1, 1)) + conv.bias)
    }
}

