# Pytorch CNN feature extraction
### Jake Lee

## Dependencies
- [PyTorch](https://pytorch.org/)
- Written for python3, but python2 compatible

## Usage
`usage: python alexnet-extraction.py dataset_dir out_dir layer batch`

### Parameters
- `dataset_dir` is the directory of the image dataset. Since this is fed directly into pytorch's [ImageFolder](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder), it expects images to be in subdirectories within this parent directory. If there is no existing subdirectory structure, simply adding another directory level will work.
- `out_dir` is the directory in which the csv file will be saved. If this directory doesn't already exist, one will be created.
- `layer` is the layer from which features will be extracted. Valid layers are `fc6`, `fc7`, and `fc8`. Features from `fc6` and `fc7` have dimension 4096, whereas features from `fc8` have dimension 1000.
- `batch` is the batch size with which images will be fed through the network. These sizes are typically in powers of 2, and a higher batch size can improve performance. However, the batch size is limited by the VRAM available in the GPU.

### Output
Output CSVs are saved to `out_dir/layer.csv`. The first value of each row is the image filename, and the rest of the values are the extracted features. 

### DEMUD Usage
Output CSV files can be used with DEMUD with the `-v` or `--cnn` option.

## Extension

### Other models
Adjusting this script for extracting from other models is straightforward. Replace `model = models.alexnet(pretrained=True)` on line 58 with the preferred pretrained model [offered by Pytorch](https://pytorch.org/docs/stable/torchvision/models.html). Hooks defined from lines 62 to 85 will have to be modified (for VGG16, for example) or removed (for ResNet-50, for example). 

### Finetuned models
If you want to use non-pretrained models (for example, the [anti-aliased models by Richard Zhang](https://richzhang.github.io/antialiased-cnns/)), line 58 needs to be replaced by the architecture definition, and weights should be loaded via `model.load_state_dict(torch.load('weights.pth.tar')['state_dict'])`.
