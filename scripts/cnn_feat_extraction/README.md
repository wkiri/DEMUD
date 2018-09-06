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

### 1. Preprocessing

Convert images to size 227x227 pixels.  For example, using ImageMagick:

`$ mogrify -path imageset/# -format jpg -resize "227x227^" -gravity center -crop 227x227+0+0 +repage imageset/#/*.jpg`

Performing this preprocessing separately is strongly recommended, as the caffe resizing can be quite slow.

### 2. Configuration File

Create a config file to specify locations of the trained CNN (Caffe model), the image files, and other parameters.  You can use `example.cfg` as a starting point.

*[Files]*
- `model_dir` is the folder where caffe model definitions are stored.
- `imageset_dir` is the imageset directory. It contains images or subdirectories. Do not include non-image files in the directory; while they will be rejected, the imagecount will be incorrect.
- `model_def` is the `.prototxt` file within `imageset_dir`.
- `model_weights` is the `.caffemodel` file within `imageset_dir`.
- `mean_image` is the `.npy` file within `imageset_dir`.
- `out_dir` is the directory in which the extracted features will be stored. If it doesn't exist, it will be created.

*[Params]*
- `batch_size` is the batch size for the neural network. If on CPU, the recommended value is `1`. If on GPU, it can be increased to `16` or greater for a faster runtime.
- `GPU_enable` defines whether to run caffe in CPU or GPU modes. `0` for CPU, `1` for GPU.
- `layer_list` defines which layers to extract features from. Layer names are listed in the `.prototxt` file that comes with the pretrained model.
- `class_list` defines how to deal with different imageset structures.
    - `<all>` tells the script to look for images in all subdirectories. For this option, `qty_list` must also be defined as `<all>`.
	- `<flat>` tells the script that there are no subdirectories; all of the images are in the top level of `imageset_dir`. `qty_list` can only be one value.
	- `dir1,dir2,dir3` tells the script to look in the specified subdirectories for the images. `qty_list` must be the same length.
- `qty_list` defines how many images to extract features from.
    - `<all>` tells the script to extract from all images in the subdirectories specified.
	- `10` tells the script to extract 10 images from `<flat>`, or the single subdirectory specified.
	- `10,20,30` tells the script to extract 10, 20, and 30 images from the specified subdirectories, respectively.
	- `10,all,30` - the `all` keyword can be mixed within the list to specify that all images in the respective subdirectory are to be extracted from.

### 3. Feature Extraction

Extract the CNN feature vectors, which will be saved to .csv files:

`python feat_csv.py your_config_file.cfg`

The first column is a label of the image's subdirectory (if it exists) and the filename as `[subdir]-[filename]`. If the file was deemed not an image, the label will have `BAD` appended to it. 

### 4. DEMUD usage

The resulting .csv files can be read in by DEMUD with the `-v` or `--cnn` option.

