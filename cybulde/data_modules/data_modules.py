from typing import Any, Callable, Optional, Protocol

from lightning.pytorch import LightningDataModule
from torch import Tensor
from torch.utils.data import BatchSampler, DataLoader, Dataset, Sampler, default_collate
from transformers import BatchEncoding

from cybulde.data_modules.datasets import TextClassificationDataset
# from cybulde.models.transformations import HuggingFaceTokenizationTransformation, Transformation


# Create a base class for each Data Module.
class DataModule(LightningDataModule):
    # Parameters we need for Data Loaders
    #  doc.: https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    def __init__(
        self,
        batch_size: int,
        shuffle: bool = False,
        sampler: Optional[Sampler] = None,
        batch_sampler: Optional[BatchSampler] = None,
        num_workers: int = 0,
        collate_fn: Optional[Callable[[Any], Any]] = None, # Function that is going tobe run after your data loader constructs a batch.
        pin_memory: bool = False, # Related to GPU. When True the samples read will be directly copied to GPU memory.
        drop_last: bool = False, # set to True to drop the last incomplete batch, if the dataset size is not divisible by the batch size.
        persistent_workers: bool = False, # If True, the data loader will not shut down the worker processes after a dataset has been consumed once.
    ) -> None:
        super().__init__()

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sampler = sampler
        self.batch_sampler = batch_sampler
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.persistent_workers = persistent_workers

    # 
    def initialize_dataloader(self, dataset: Dataset, is_test: bool) -> DataLoader: # is_test = True for development, validation or test datasets, and = False for training dataset.
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle and not is_test,
            sampler=self.sampler,
            batch_sampler=self.batch_sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            persistent_workers=self.persistent_workers,
        )


# class PartialDataModuleType(Protocol):
#     def __call__(self, transformation: Transformation) -> DataModule:
#         ...


class TextClassificationDataModule(DataModule):
    def __init__(
        self,
        train_df_path: str,
        dev_df_path: str,
        test_df_path: str,
        transformation: HuggingFaceTokenizationTransformation,
        text_column_name: str,
        label_column_name: str,
        batch_size: int,
        shuffle: bool = False,
        sampler: Optional[Sampler] = None,
        batch_sampler: Optional[BatchSampler] = None,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False,
        persistent_workers: bool = False,
    ) -> None:
        # Collate function that takes a list of text and converts to encodings using the tokenizer we trained.
        def tokenization_collate_fn(batch: list[tuple[str, int]]) -> tuple[BatchEncoding, Tensor]:
            texts, labels = default_collate(batch) # Proccess the sample and return something we can work with.
            encodings = transformation(texts) # Call the tokenizer
            return encodings, labels

        super().__init__(
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=tokenization_collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            persistent_workers=persistent_workers,
        )

        self.train_df_path = train_df_path
        self.dev_df_path = dev_df_path
        self.test_df_path = test_df_path

        self.text_column_name = text_column_name
        self.label_column_name = label_column_name

    # When you are using a DataModule from LightningDataModule you need to define 4 more methods: setup, train_dataloader, val_dataloader and test_dataloader.
    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None: # It means we are in training stage, so we need to setup our training and validation datasets.
            self.train_dataset = TextClassificationDataset(
                self.train_df_path, self.text_column_name, self.label_column_name
            )
            self.dev_dataset = TextClassificationDataset(
                self.dev_df_path, self.text_column_name, self.label_column_name
            )

        if stage == "test": # It means we are in test stage, so we need to setup our testn dataset.
            self.test_dataset = TextClassificationDataset(
                self.test_df_path, self.text_column_name, self.label_column_name
            )

    # Initialize Data Loaders
    def train_dataloader(self) -> DataLoader:
        return self.initialize_dataloader(self.train_dataset, is_test=False)

    def val_dataloader(self) -> DataLoader:
        return self.initialize_dataloader(self.dev_dataset, is_test=True)

    def test_dataloader(self) -> DataLoader:
        return self.initialize_dataloader(self.test_dataset, is_test=True)