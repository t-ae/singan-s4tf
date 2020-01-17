import Foundation
import Swim
import TensorFlow

struct Size: CustomStringConvertible {
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
    
    var description: String {
        "\(width)x\(height)"
    }
}

struct ImagePyramid {
    // coarse to fine
    let images: [Tensor<Float>]
    
    var sizes: [Size] {
        images.map { Size(width: $0.shape[2], height: $0.shape[1]) }
    }
    
    init(images: [Tensor<Float>]) {
        self.images = images
    }
    
    subscript(i: Int) -> Tensor<Float> {
        return images[i]
    }
    
    static func load(file: URL, config: Config) throws -> ImagePyramid {
        let baseImage = try Image<RGB, Float>(contentsOf: file)
        
        // keep aspect ratio
        let maxSize = Size(width: baseImage.width, height: baseImage.height)
            .fit(maxSize: config.imageMaxSize)
        
        var sizeList: [Size] = [maxSize]
        while true {
            let currentSize = sizeList.first!
            let newSize = currentSize.scaled(factor: config.scaleFactor)
            guard newSize.width >= config.imageMinSize && newSize.height >= config.imageMinSize else {
                break
            }
            sizeList.insert(newSize, at: 0)
        }
        
        let images: [Image<RGB, Float>] = sizeList.map { size in
            baseImage.resize(width: size.width, height: size.height, method: .areaAverage)
        }
        
        return ImagePyramid(images: images.map {
            2 * Tensor(image: $0).expandingShape(at: 0) - 1
        })
    }
}

