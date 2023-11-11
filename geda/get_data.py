from typing import Literal
import geda.data_providers as gdp

DATA_PROVIDERS = {
    "MNIST": gdp.mnist.MNISTDataProvider,
    "DUTS": gdp.duts.DUTSDataProvider,
    "NYUDv2": gdp.nyud.NYUDv2DataProvider,
    "VOC_InstanceSegmentation": gdp.voc.VOCInstanceSegmentationDataProvider,
    "VOC_SemanticSegmentation": gdp.voc.VOCSemanticSegmentationDataProvider,
    "VOC_PersonPartSegmentation": gdp.voc.VOCPersonPartSegmentationDataProvider,
    "VOC_SemanticSegmentationAug": gdp.voc.VOCSemanticSegmentationAugDataProvider,
    "VOC_Main": gdp.voc.VOCMainDataProvider,
    "VOC_Action": gdp.voc.VOCActionDataProvider,
    "VOC_Layout": gdp.voc.VOCLayoutDataProvider,
    "MPII": gdp.mpii.MPIIDataProvider,
    "COCO_Keypoints": gdp.coco.COCOKeypointsDataProvider,
}

DATA_PROVIDERS_NAMES = Literal[
    "MNIST",
    "DUTS",
    "NYUDv2",
    "VOC_InstanceSegmentation",
    "VOC_SemanticSegmentation",
    "VOC_SemanticSegmentationAug",
    "VOC_PersonPartSegmentation",
    "VOC_Main",
    "VOC_Action",
    "VOC_Layout",
    "MPII",
    "COCO_Keypoints",
]


def get_data(name: DATA_PROVIDERS_NAMES, root: str, **kwargs) -> gdp.base.DataProvider:
    data_provider: gdp.base.DataProvider = DATA_PROVIDERS[name](root=root)
    data_provider.get_data()
    return data_provider
