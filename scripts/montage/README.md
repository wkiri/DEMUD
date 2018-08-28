# Montage generation
### Kiri Wagstaff

## Dependencies
* [ImageMagick](https://www.imagemagick.org/script/index.php) 

## Usage

`montage.py [-h] [-n NUMSEL] [-o OUTDIR] selectionsfile imagedir`

This script creates a montage image that contains the first `NUMSEL`
(defaults to all) images included in the specified DEMUD
selectionsfile (`selections-k*.csv`).  The image files are read from
`imagedir`, and the resulting `montage-k*.jpg` file is written to
`OUTDIR` (defaults to the current working directory).

If the number of images to be included in the montage is greater than 100,
the script will offer to truncate to the first 100 (but allow you to
continue if desired).


