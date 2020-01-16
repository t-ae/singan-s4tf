import TensorFlow

protocol Loss {
    var name: String { get }
    @differentiable
    func lossG(_ tensor: Tensor<Float>) -> Tensor<Float>
    
    @differentiable
    func lossD(real: Tensor<Float>, fake: Tensor<Float>) -> Tensor<Float>
}

struct NonSaturatingLoss: Loss {
    let name: String = "NonSaturating"
    
    @differentiable
    func lossG(_ tensor: Tensor<Float>) -> Tensor<Float> {
        softplus(-tensor).mean()
    }

    @differentiable
    func lossD(real: Tensor<Float>, fake: Tensor<Float>) -> Tensor<Float> {
        softplus(-real).mean() + softplus(fake).mean()
    }
}

struct LSGANLoss: Loss {
    let name = "LSGAN"
    
    @differentiable
    func lossG(_ tensor: Tensor<Float>) -> Tensor<Float> {
        pow(tensor - 1, 2).mean()
    }

    @differentiable
    func lossD(real: Tensor<Float>, fake: Tensor<Float>) -> Tensor<Float> {
        pow(real-1, 2).mean() + pow(fake, 2).mean()
    }
}

struct HingeLoss: Loss {
    let name = "Hinge"
    
    @differentiable
    func lossG(_ tensor: Tensor<Float>) -> Tensor<Float> {
        -tensor.mean()
    }

    @differentiable
    func lossD(real: Tensor<Float>, fake: Tensor<Float>) -> Tensor<Float> {
        relu(1 - real).mean() + relu(1 + fake).mean()
    }
}
