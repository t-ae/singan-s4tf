import TensorFlow

protocol GANLoss {
    @differentiable
    func lossG(_ tensor: Tensor<Float>) -> Tensor<Float>
    
    @differentiable
    func lossD(real: Tensor<Float>, fake: Tensor<Float>) -> Tensor<Float>
}

struct NonSaturatingLoss: GANLoss {
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

struct LSGANLoss: GANLoss {
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

struct HingeLoss: GANLoss {
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

protocol ReconstructionLoss {
    @differentiable(wrt: fake)
    func callAsFunction(real: Tensor<Float>, fake: Tensor<Float>) -> Tensor<Float>
}

struct MSELoss: ReconstructionLoss {
    let name = "MSE"
    
    @differentiable(wrt: fake)
    func callAsFunction(real: Tensor<Float>, fake: Tensor<Float>) -> Tensor<Float> {
        meanSquaredError(predicted: fake, expected: real)
    }
}

struct BCELoss: ReconstructionLoss {
    let name = "BCE"
    
    @differentiable(wrt: fake)
    func callAsFunction(real: Tensor<Float>, fake: Tensor<Float>) -> Tensor<Float> {
        // [0,1] range
        var real = (real + 1) / 2
        real = real.clipped(min: 0, max: 1)
        var fake = (fake + 1) / 2
        fake = fake.clipped(min: 0, max: 1)
        
        let loss1 = -(real) * log(fake)
        let loss2 = -(1-real) * log(1-fake)
        return (loss1 + loss2).mean()
    }
}
