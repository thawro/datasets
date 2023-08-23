from typing import Literal
from geda import data_providers

DATA_PROVIDERS = {
    "DUTS": data_providers.duts.DUTSDataProvider,
    "NYUDv2": data_providers.nyud.NYUDv2DataProvider,
    "VOC_InstanceSegmentation": data_providers.voc.VOCInstanceSegmentationDataProvider,
    "VOC_SemanticSegmentation": data_providers.voc.VOCSemanticSegmentationDataProvider,
    "VOC_PersonPartSegmentation": data_providers.voc.VOCPersonPartSegmentationDataProvider,
    "VOC_Main": data_providers.voc.VOCMainDataProvider,
    "VOC_Action": data_providers.voc.VOCActionDataProvider,
    "VOC_Layout": data_providers.voc.VOCLayoutDataProvider,
}

DATA_PROVIDERS_NAMES = Literal[
    "DUTS",
    "NYUDv2",
    "VOC_InstanceSegmentation",
    "VOC_SemanticSegmentation",
    "VOC_PersonPartSegmentation",
    "VOC_Main",
    "VOC_Action",
    "VOC_Layout",
]


def get_data(name: DATA_PROVIDERS_NAMES, root: str, **kwargs) -> data_providers.base.DataProvider:
    data_provider: data_providers.base.DataProvider = DATA_PROVIDERS[name](root=root)
    data_provider.get_data()
    return data_provider
