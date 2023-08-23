# GeDa

**GeDa** is a Python package that helps you to **Ge**t the **Da**ta for your project.

## Installation

```bash
pip install geda
```

## Usage

### Using specific data provider class

```python
from geda.data_providers.voc import VOCSemanticSegmentationDataProvider

root = "<directory>/<to>/<store>/<data>" # e.g. "data/VOC"
dataprovider = VOCSemanticSegmentationDataProvider(root)
dataprovider.get_data()
```

### Using `get_data` shortcut

```python
from geda import get_data

root = "<directory>/<to>/<store>/<data>" # e.g. "data/VOC"
dataprovider = get_data(name="VOC_SemanticSegmentation", root=root)
dataprovider.get_data()
```

> The `get_data` function currently supported names:
> `DUTS`, `NYUDv2`, `VOC_InstanceSegmentation`, `VOC_SemanticSegmentation`, `VOC_PersonPartSegmentation`, `VOC_Main`, `VOC_Action`, `VOC_Layout`


## What it does

By using `dataprovider.get_data()` functionality, the data is subjected to the following pipeline:

1. Download the data from source (specified by the `_URLS` variable in each module)
2. Unzip the files if needed (in case of `tar`, `zip` or `gz` files downloaded)
3. Move the files to `<root>/raw` directory
4. Find the split ids (file basenames or indices - depending on the dataset)
5. Arrange files, i.e. move (or copy) files from `<root>/raw` directory to task-specific directories
6. *[Optional]* Create labels in specific format (f.e. YOLO)

### Example

Resulting directory structure of the `get_data(name="VOC_SemanticSegmentation", root="data/VOC")`

    .
    └── data
        └── VOC
            ├── raw
            │   ├── Annotations
            │   ├── ImageSets
            │   ├── JPEGImages
            │   ├── SegmentationClass
            │   └── SegmentationObject
            ├── SegmentationClass
            │   ├── annots
            │   ├── images
            │   ├── labels
            │   └── masks
            └── trainval_2012.tar

## Currently supported datasets

### Image Segmentation

* [PASCAL VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC)
* [NYUDv2](https://cs.nyu.edu/~silberman/projects/indoor_scene_seg_sup.html)
* [Person-Parts](http://liangchiehchen.com/projects/DeepLab.html)
* [DUTS](http://saliencydetection.net/duts/)


## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.


## License

[MIT](https://choosealicense.com/licenses/mit/)