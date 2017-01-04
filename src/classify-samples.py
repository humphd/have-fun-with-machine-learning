# python classify-samples.py -c ~/repos/caffe/ -m ../data/googlenet-dolphins-and-seahorses

import numpy as np
import os
import sys
import argparse

class ImageClassifier:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        deploy_file = os.path.join(model_dir, 'deploy.prototxt')
        weights_file = os.path.join(model_dir, 'snapshot_iter_90.caffemodel')
        self.net = caffe.Net(deploy_file, caffe.TEST, weights=weights_file)

    def setup(self):
        mean_file = os.path.join(self.model_dir, 'mean.binaryproto')
        labels_file = os.path.join(self.model_dir, 'labels.txt')

        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2, 0, 1))
        self.transformer.set_raw_scale('data', 255)
        self.transformer.set_channel_swap('data', (2, 1, 0))

        # set mean pixel
        with open(mean_file, 'rb') as infile:
            blob = caffe_pb2.BlobProto()
            blob.MergeFromString(infile.read())
            if blob.HasField('shape'):
                blob_dims = blob.shape
                assert len(blob_dims) == 4, 'Shape should have 4 dimensions - shape is %s' % blob.shape
            elif blob.HasField('num') and blob.HasField('channels') and \
                    blob.HasField('height') and blob.HasField('width'):
                blob_dims = (blob.num, blob.channels, blob.height, blob.width)
            else:
                raise ValueError('blob does not provide shape or 4d dimensions')
            pixel = np.reshape(blob.data, blob_dims[1:]).mean(1).mean(1)
            self.transformer.set_mean('data', pixel)

        # This is overkill here, since we only have 2 labels, but here's how we might read them.
        # Later we'd grab the label we want based on position (e.g., 0=dolphin, 1=seahorse)
        self.labels = np.loadtxt(labels_file, str, delimiter='\n')

    def classify(self, fullpath):
        # Load the image from disk using caffe's built-in I/O module
        image = caffe.io.load_image(fullpath)
        # Preprocess the image into the proper format for feeding into the model
        self.net.blobs['data'].data[...] = self.transformer.preprocess('data', image)
        # Run the image's pixel data through the network
        out = self.net.forward()
        # Extract the probabilities of our two categories from the final layer
        softmax_layer = out['softmax']
        # Here we're converting to Python types from ndarray floats
        dolphin_prob = softmax_layer.item(0)
        seahorse_prob = softmax_layer.item(1)

        # Print the results. I'm using labels
        label = self.labels[0] if dolphin_prob > seahorse_prob else self.labels[1]
        filename = os.path.basename(fullpath)
        print '%s is a %s dolphin=%.3f%% seahorse=%.3f%%' % (filename, label, dolphin_prob*100, seahorse_prob*100)

def setup_caffe(caffe_root):
    # Load Caffe's Python interface from the specified path
    sys.path.insert(0, os.path.join(caffe_root, 'python'))
    global caffe
    global caffe_pb2
    import caffe
    from caffe.proto import caffe_pb2

    # Set Caffe to use CPU mode so this will work on as many machines as possible.
    caffe.set_mode_cpu()

def main():
    parser = argparse.ArgumentParser(
        description='Classify images of dolphins and seahorses using trained Caffe model'
    )
    parser.add_argument('-c', '--caffe_root', help='CAFFE_ROOT dir, if not defined in env')
    parser.add_argument('-m', '--model_dir', help='Trained model dir, downloaded from DIGITS')
    parser.add_argument('-d', '--images_dir', help='Directory of images to classify')

    args = parser.parse_args()

    # Prefer $CAFFE_ROOT in the env if it exists, otherwise get from args
    caffe_root = os.getenv('CAFFE_ROOT') or args.caffe_root
    if not caffe_root:
        print 'Error: Missing CAFFE_ROOT dir. Set env variable or pass via --caffe_root'
        parser.print_help()
        sys.exit(1)
    setup_caffe(caffe_root)
    
    model_dir = args.model_dir
    if not model_dir or not os.path.isdir(model_dir):
        print 'Error: Unable to find model files. Pass dir via --model_dir'
        parser.print_help()
        sys.exit(1)
    classifier = ImageClassifier(model_dir)
    classifier.setup()

    # Allow passing images dir, or use ../data/untrained-samples by default
    cwd = os.path.dirname(os.path.abspath(__file__))
    untrained_samples = os.path.join(cwd, '..', 'data', 'untrained-samples')
    images_dir = args.images_dir or untrained_samples
    if not os.path.isdir(images_dir):
        print 'Error: Unable to find images for classification. Pass dir via --images_dir'
        parser.print_help()        
        sys.exit(1)
    
    # Classify all images in images_dir using our trained network
    for filename in os.listdir(images_dir):
        if filename.endswith('.jpg'):
            classifier.classify(os.path.join(images_dir, filename))

if __name__ == '__main__':
    main()
