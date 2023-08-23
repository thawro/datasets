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