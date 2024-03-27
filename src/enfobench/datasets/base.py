from pathlib import Path


class DatasetBase:
    SUBSETS = ()

    def __init__(self, directory: Path | str) -> None:
        directory = Path(directory).resolve()
        if not directory.is_dir() or not directory.exists():
            msg = "Please provide an existing directory where the dataset is located."
            raise ValueError(msg)
        self.directory = directory

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(directory={self.directory})"

    def _check_for_valid_subset(self, subset: str):
        if subset not in self.SUBSETS:
            msg = f"Please provide a valid subset. Available subsets: {self.SUBSETS}"
            raise ValueError(msg)

    def _get_subset_path(self, subset: str, extension: str = "parquet") -> Path:
        filepath = self.directory / f"{subset}.{extension}"
        if not filepath.exists():
            msg = f"Subset: {subset} is missing from the directory."
            raise ValueError(msg)
        return filepath
