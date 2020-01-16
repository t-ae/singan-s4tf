import TensorBoardX

extension SNConv2D: HistogramWritable where Scalar == Float {
    public func writeHistograms(tag: String, writer: SummaryWriter, globalStep: Int?) {
        writer.addHistogram(tag: "\(tag).filter", values: conv.filter, globalStep: globalStep)
        writer.addHistogram(tag: "\(tag).bias", values: conv.bias, globalStep: globalStep)
    }
}

extension ConvBlock: HistogramWritable {
    public func writeHistograms(tag: String, writer: SummaryWriter, globalStep: Int?) {
        conv.writeHistograms(tag: tag, writer: writer, globalStep: globalStep)
    }
}

extension Generator: HistogramWritable {
    public func writeHistograms(tag: String, writer: SummaryWriter, globalStep: Int?) {
        head.writeHistograms(tag: "\(tag).head", writer: writer, globalStep: globalStep)
        conv1.writeHistograms(tag: "\(tag).conv1", writer: writer, globalStep: globalStep)
        conv2.writeHistograms(tag: "\(tag).conv2", writer: writer, globalStep: globalStep)
        conv3.writeHistograms(tag: "\(tag).conv3", writer: writer, globalStep: globalStep)
        tail.writeHistograms(tag: "\(tag).tail", writer: writer, globalStep: globalStep)
    }
}

extension Discriminator: HistogramWritable {
    public func writeHistograms(tag: String, writer: SummaryWriter, globalStep: Int?) {
        head.writeHistograms(tag: "\(tag).head", writer: writer, globalStep: globalStep)
        conv1.writeHistograms(tag: "\(tag).conv1", writer: writer, globalStep: globalStep)
        conv2.writeHistograms(tag: "\(tag).conv2", writer: writer, globalStep: globalStep)
        conv3.writeHistograms(tag: "\(tag).conv3", writer: writer, globalStep: globalStep)
        tail.writeHistograms(tag: "\(tag).tail", writer: writer, globalStep: globalStep)
    }
}
