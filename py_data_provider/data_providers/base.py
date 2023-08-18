from py_data_provider.utils.files import download_file, unzip_tar_gz, path_exists
from py_data_provider.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class TarDataProvider:
    def __init__(self, urls: dict[str, str], root: str):
        self.urls = urls
        self.root = root
        self.raw_root = f"{root}/raw"
        self.tar_filepaths = [f"{root}/{name}.tar.gz" for name in urls]
        if not self.check_if_present():
            self.download()  # TODO
            self.unzip()
            self.set_split_ids()
            self.arrange_files()
            self.create_labels()
            self.set_filepaths()
        else:
            log.info(f"Dataset is already present")
            self.set_split_ids()
            self.set_filepaths()

    def check_if_present(self) -> bool:
        raise NotImplementedError()

    def download(self):
        for name, url in self.urls.items():
            tar_filepath = f"{self.root}/{name}.tar.gz"
            if path_exists(tar_filepath):
                log.info(f"{tar_filepath} is already present. Download canceled")
                return
            download_file(url, tar_filepath)

    def unzip(self, remove: bool = False):
        if path_exists(self.raw_root):
            log.info(f"{self.raw_root} is already present. Unzip canceled")
            return
        for tar_filepath in self.tar_filepaths:
            unzip_tar_gz(tar_filepath, self.root, remove=remove)
        self.move_to_raw_root()
        if remove:
            self.tar_filepaths.clear()

    def move_to_raw_root(self):
        raise NotImplementedError()

    def set_split_ids(self):
        raise NotImplementedError()

    def arrange_files(self):
        raise NotImplementedError()

    def set_filepaths(self):
        raise NotImplementedError()

    def create_labels(self):
        raise NotImplementedError()
