# CNN image feature extraction
### Kiri Wagstaff, Jake Lee
### in WHI 2018

## Dependencies
* [Caffe](http://caffe.berkeleyvision.org/)
* [CaffeNet Model](https://github.com/BVLC/caffe/tree/master/models/bvlc_reference_caffenet)
* [NumPy](http://www.numpy.org/)
* [ImageMagick](https://www.imagemagick.org/script/index.php) (optional)

## Usage

`python feat_csv.py example.cfg` exports the feature vectors of the specified images from the specified neural net layers as a csv. This csv can then be input into DEMUD with the `-v` or `--cnn` option.

`example.cfg` specifies all information required, such as the location of the caffe model files, the location of the imageset, and the number of classes and images to export. The example configuration contains more instructions on the options available.

Note that the images must be preprocessed to 227x227 before extracting its features. For example, for our experiments, we used imagemagick:
`$ mogrify -path imageset/# -format jpg -resize "227x227^" -gravity center -crop 227x227+0+0 +repage imageset/#/*.jpg`

A future update will perform this preprocessing within the script.