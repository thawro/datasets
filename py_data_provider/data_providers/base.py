from py_data_provider.utils.files import download_file, unzip_tar, path_exists, unzip_zip
from py_data_provider.utils.pylogger import get_pylogger
from abc import abstractmethod
from pathlib import Path

log = get_pylogger(__name__)


class DataProvider:
    def __init__(self, urls: dict[str, str], root: str):
        self.urls = urls
        self.root = root
        self.raw_root = f"{root}/raw"
        if not self.check_if_present():
            self.zip_filepaths = self.download()  # TODO
            self.unzip()
            self.split_ids = self._get_split_ids()
            self.arrange_files()
            self.create_labels()
            self.filepaths = self._get_filepaths()
        else:
            log.info(f"Dataset is already present")
            self.split_ids = self._get_split_ids()
            self.filepaths = self._get_filepaths()

    def download(self):
        zip_filepaths = []
        for name, url in self.urls.items():
            ext = url.split(".")[-1]
            zip_filepath = f"{self.root}/{name}.{ext}"
            zip_filepaths.append(zip_filepath)
            if path_exists(zip_filepath):
                log.info(f"{zip_filepath} is already present. Download canceled")
            else:
                download_file(url, zip_filepath)
        return zip_filepaths

    def unzip(self, remove: bool = False):
        if path_exists(self.raw_root):
            log.info(f"{self.raw_root} is already present. Unzip canceled")
            return
        Path(self.raw_root).mkdir(parents=True, exist_ok=True)
        for zip_filepath in self.zip_filepaths:
            self._unzip(zip_filepath, remove=remove)
        self.move_to_raw_root()
        if remove:
            self.zip_filepaths.clear()

    @abstractmethod
    def _unzip(self, file_path: str, remove: bool = False):
        raise NotImplementedError()

    @abstractmethod
    def check_if_present(self) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def move_to_raw_root(self):
        raise NotImplementedError()

    @abstractmethod
    def _get_split_ids(self):
        raise NotImplementedError()

    @abstractmethod
    def arrange_files(self):
        raise NotImplementedError()

    @abstractmethod
    def _get_filepaths(self):
        raise NotImplementedError()

    @abstractmethod
    def create_labels(self):
        raise NotImplementedError()


class TarDataProvider(DataProvider):
    def _unzip(self, file_path: str, remove: bool = False):
        unzip_tar(file_path, self.root, remove=remove)


class ZipDataProvider(DataProvider):
    def _unzip(self, file_path: str, remove: bool = False):
        unzip_zip(file_path, self.root, remove=remove)
