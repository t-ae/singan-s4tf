import Foundation
import Swim
import TensorFlow

struct Size {
    var width: Int
    var height: Int
    
    func fit(maxSize: Int) -> Size {
        if width > maxSize && width >= height {
            let scale = Float(maxSize) / Float(width)
            return scaled(factor: scale)
        } else if height > maxSize && height >= width {
            let scale = Float(maxSize) / Float(height)
            return scaled(factor: scale)
        }
        return self
    }
    
    func scaled(factor: Float) -> Size {
        Size(width: Int(Float(width)*factor), height: Int(Float(height)*factor))
    }
}

struct ImagePyramid {
    // coarse to fine
    let images: [Tensor<Float>]
    
    var sizes: [Size] {
        images.map { Size(width: $0.shape[1], height: $0.shape[0]) }
    }
    
    init(images: [Tensor<Float>]) {
        self.images = images
    }
    
    subscript(i: Int) -> Tensor<Float> {
        return images[i]
    }
    
    static func load(file: URL) throws -> ImagePyramid {
        let baseImage = try Image<RGB, Float>(contentsOf: file)
        
        // keep aspect ratio
        let maxSize = Size(width: baseImage.width, height: baseImage.height)
            .fit(maxSize: Config.imageMaxSize)
        
        var sizeList: [Size] = [maxSize]
        while true {
            let currentSize = sizeList.first!
            let newSize = currentSize.scaled(factor: Config.scaleFactor)
            guard newSize.width >= Config.imageMinSize && newSize.height >= Config.imageMinSize else {
                break
            }
            sizeList.insert(newSize, at: 0)
        }
        
        let images: [Image<RGB, Float>] = sizeList.map { size in
            baseImage.resize(width: size.width, height: size.height, method: .bilinear)
        }
        
        return ImagePyramid(images: images.map {
            Tensor(shape: [$0.height, $0.width, 3], scalars: $0.getData())
        })
    }
}

