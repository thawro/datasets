from typing import Literal
from geda import data_providers
import geda.data_providers as gdp

DATA_PROVIDERS = {
    "DUTS": gdp.duts.DUTSDataProvider,
    "NYUDv2": gdp.nyud.NYUDv2DataProvider,
    "VOC_InstanceSegmentation": gdp.voc.VOCInstanceSegmentationDataProvider,
    "VOC_SemanticSegmentation": gdp.voc.VOCSemanticSegmentationDataProvider,
    "VOC_PersonPartSegmentation": gdp.voc.VOCPersonPartSegmentationDataProvider,
    "VOC_SemanticSegmentationAug": gdp.voc.VOCSemanticSegmentationAugDataProvider,
    "VOC_Main": gdp.voc.VOCMainDataProvider,
    "VOC_Action": gdp.voc.VOCActionDataProvider,
    "VOC_Layout": gdp.voc.VOCLayoutDataProvider,
}

DATA_PROVIDERS_NAMES = Literal[
    "DUTS",
    "NYUDv2",
    "VOC_InstanceSegmentation",
    "VOC_SemanticSegmentation",
    "VOC_SemanticSegmentationAug",
    "VOC_PersonPartSegmentation",
    "VOC_Main",
    "VOC_Action",
    "VOC_Layout",
]


def get_data(name: DATA_PROVIDERS_NAMES, root: str, **kwargs) -> gdp.base.DataProvider:
    data_provider: gdp.base.DataProvider = DATA_PROVIDERS[name](root=root)
    data_provider.get_data()
    return data_provider
