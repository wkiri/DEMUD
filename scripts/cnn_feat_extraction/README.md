# CNN image feature extraction
### Kiri Wagstaff, Jake Lee
### in WHI 2018

## Dependencies
* [Caffe](http://caffe.berkeleyvision.org/)
* [CaffeNet Model](https://github.com/BVLC/caffe/tree/master/models/bvlc_reference_caffenet)
* [NumPy](http://www.numpy.org/)
* [ImageMagick](https://www.imagemagick.org/script/index.php) (optional)

## Usage

To extract the CNN feature vectors for a set of images:

1. Convert images to size 227x227 pixels.  For example, using ImageMagick:

`$ mogrify -path imageset/# -format jpg -resize "227x227^" -gravity center -crop 227x227+0+0 +repage imageset/#/*.jpg`

A future update will perform this preprocessing within the `feat_csv.py` script.

2. Create a config file to specify locations of the trained CNN (Caffe model), the image files, and other parameters.  You can use `example.cfg` as a starting point.

3. Extract the CNN feature vectors, which will be saved to .csv files:

`python feat_csv.py your_config_file.cfg` 

4. The resulting .csv files can be read in by DEMUD with the `-v` or `--cnn` option.

