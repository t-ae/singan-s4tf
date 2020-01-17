import TensorFlow

enum GANLossType: String, Codable {
    case nonSaturating, lsgan, hinge
}

struct GANLoss {
    let type: GANLossType
    
    @differentiable
    func lossG(_ tensor: Tensor<Float>) -> Tensor<Float> {
        switch type {
        case .nonSaturating:
            return softplus(-tensor).mean()
        case .lsgan:
            return pow(tensor - 1, 2).mean()
        case .hinge:
            return -tensor.mean()
        }
    }
    
    @differentiable
    func lossD(real: Tensor<Float>, fake: Tensor<Float>) -> Tensor<Float> {
        switch type {
        case .nonSaturating:
            return softplus(-real).mean() + softplus(fake).mean()
        case .lsgan:
            return pow(real-1, 2).mean() + pow(fake, 2).mean()
        case .hinge:
            return relu(1 - real).mean() + relu(1 + fake).mean()
        }
    }
}

enum ReconstructionLossType: String, Codable {
    case meanSquaredError, binaryCrossEntropy
}

struct ReconstructionLoss {
    let type: ReconstructionLossType
    
    @differentiable(wrt: fake)
    func callAsFunction(real: Tensor<Float>, fake: Tensor<Float>) -> Tensor<Float> {
        switch type {
        case .meanSquaredError:
            return meanSquaredError(predicted: fake, expected: real)
        case .binaryCrossEntropy:
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
}
