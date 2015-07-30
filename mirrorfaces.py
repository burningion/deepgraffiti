from cStringIO import StringIO
import numpy as np
import scipy.ndimage as nd
import PIL.Image
from IPython.display import clear_output, Image, display
from google.protobuf import text_format

import caffe
caffe.set_mode_gpu()
def showarray(a, fmt='jpeg'):
    a = np.uint8(np.clip(a, 0, 255))
    f = StringIO()
    PIL.Image.fromarray(a).save(f, fmt)
    #display(Image(data=f.getvalue()))

model_path = '../caffe/models/faces_net/' # substitute your path here
mean_filename= model_path + 'mean.binaryproto'
proto_data = open(mean_filename, "rb").read()
a = caffe.io.caffe_pb2.BlobProto.FromString(proto_data)
mean  = caffe.io.blobproto_to_array(a)[0]
net_fn   = model_path + 'deploy_gender.prototxt'
param_fn = model_path + 'gender_net.caffemodel'

# Patching model to be able to compute gradients.
# Note that you can also manually add "force_backward: true" line to "deploy.prototxt".
model = caffe.io.caffe_pb2.NetParameter()
text_format.Merge(open(net_fn).read(), model)
model.force_backward = True
open('tmp.prototxt', 'w').write(str(model))

net = caffe.Classifier('tmp.prototxt', param_fn,
                       mean = mean.mean(1).mean(1), #mean, # ImageNet mean, training set dependent
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256)) # the reference model has channels in BGR order instead of RGB

# a couple of utility functions for converting to and from Caffe's input image layout
def preprocess(net, img):
    return np.float32(np.rollaxis(img, 2)[::-1]) - net.transformer.mean['data']
def deprocess(net, img):
    return np.dstack((img + net.transformer.mean['data'])[::-1])

def make_step(net, step_size=1.5, end='fc8_oxford_102', jitter=32, clip=True):
    '''Basic gradient ascent step.'''

    src = net.blobs['data'] # input image is stored in Net's 'data' blob
    dst = net.blobs[end]

    ox, oy = np.random.randint(-jitter, jitter+1, 2)
    src.data[0] = np.roll(np.roll(src.data[0], ox, -1), oy, -2) # apply jitter shift
            
    net.forward(end=end)
    dst.diff[:] = dst.data  # specify the optimization objective
    net.backward(start=end)
    g = src.diff[0]
    # apply normalized ascent step to the input image
    src.data[:] += step_size/np.abs(g).mean() * g

    src.data[0] = np.roll(np.roll(src.data[0], -ox, -1), -oy, -2) # unshift image
            
    if clip:
        bias = net.transformer.mean['data']
        src.data[:] = np.clip(src.data, -bias, 255-bias)

def deepdream(net, base_img, iter_n=10, octave_n=4, octave_scale=1.4, end='fc8_oxford_102', clip=True, **step_params):
    # prepare base images for all octavese
    octaves = [preprocess(net, base_img)]
    for i in xrange(octave_n-1):
        octaves.append(nd.zoom(octaves[-1], (1, 1.0/octave_scale,1.0/octave_scale), order=1))
    
    src = net.blobs['data']
    detail = np.zeros_like(octaves[-1]) # allocate image for network-produced details
    for octave, octave_base in enumerate(octaves[::-1]):
        h, w = octave_base.shape[-2:]
        if octave > 0:
            # upscale details from the previous octave
            h1, w1 = detail.shape[-2:]
            detail = nd.zoom(detail, (1, 1.0*h/h1,1.0*w/w1), order=1)

        src.reshape(1,3,h,w) # resize the network's input image size
        src.data[0] = octave_base+detail
        for i in xrange(iter_n):
            make_step(net, end=end, clip=clip, **step_params)
            
            # visualization
            vis = deprocess(net, src.data[0])
            if not clip: # adjust image contrast if clipping is disabled
                vis = vis*(255.0/np.percentile(vis, 99.98))
            showarray(vis)
            print octave, i, end, vis.shape
            clear_output(wait=True)
            
        # extract details produced on the current octave
        detail = src.data[0]-octave_base
    # returning the resulting image
    return deprocess(net, src.data[0])


from PIL import Image
import random

import cv2

cascPath = './haarcascade_frontalface_default.xml'

classy = cv2.CascadeClassifier(cascPath)

def get_white_noise_image(width, height):
    pil_map = Image.new("RGBA", (width, height), 255)
    random_grid = map(lambda x: (
            int(random.random() * 256),
            int(random.random() * 256),
            int(random.random() * 256)
        ), [0] * width * height)
    pil_map.putdata(random_grid)
    return pil_map

img = PIL.Image.open('face.jpg')

image = cv2.imread('face.jpg', 1)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


faces = classy.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv2.cv.CV_HAAR_SCALE_IMAGE
)
print len(faces)

facesImages = []
for (x, y, w, h) in faces:
    facesImages.append( img.crop((x-10, y-10, x+w+10, y+h+10)))
    #cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)



x = 0
y = 0
i = 0

angles = [0, 45, 90, 180, 270]

blankIMG = Image.new("RGB", (1280, 720), "white" )
while y < blankIMG.height:
    '''
    x = random.randint(0,img.width)
    y = random.randint(0,img.height)
    '''
    if x > blankIMG.width:
        y = y + facesImages[0].height
        x = 0
    if (i % 2) == 0:
        blankIMG.paste(facesImages[random.randint(0,len(facesImages)-1)], (x,y)) # random.randint(0,len(facesImages)-1) # for making face selection random
    else:
        blankIMG.paste(facesImages[random.randint(0,len(facesImages)-1)].transpose(Image.FLIP_LEFT_RIGHT), (x,y)) # random.randint(0,len(facesImages)-1)

    i+=1
    x = x + facesImages[0].width

blankIMG.save('presuccess.jpg')
    
imgnum = np.float32(blankIMG)
frame = deepdream(net, imgnum, end='pool5')
frame = deepdream(net, frame, end='pool5')
#frame = deepdream(net, frame, end='pool5')
#frame = deepdream(net, frame, end='pool5')

PIL.Image.fromarray(np.uint8(frame)).save("success.jpg")

image = cv2.imread('success.jpg', 1)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


faces = classy.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv2.cv.CV_HAAR_SCALE_IMAGE
)

print "Found {0} faces!".format(len(faces))

# Draw a rectangle around the faces

