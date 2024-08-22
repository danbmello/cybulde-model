
from torch.utils.data import DataLoader
from cybulde.data_modules.datasets import TextClassificationDataset


def main():
  df_path = "gs://cybulde-dbm/data/processed/rebalanced_splits/dev.parquet"
  text_column_name = "cleaned_text"
  label_column_name = "label"
  dev_dataset = TextClassificationDataset(
    df_path = df_path,
    text_column_name = text_column_name,
    label_column_name = label_column_name,
  )

  # print(dev_dataset)
  # print(f"{len(dev_dataset)}")

  # print(f"{dev_dataset[0]}")
  # print(f"{dev_dataset[1]}")
  # print(f"{dev_dataset[2]}")

  dev_dataloader = DataLoader(dev_dataset, batch_size=8)
  
  # for batch in dev_dataloader:
  #   print(batch)
  #   exit(0)

  for texts, labels in dev_dataloader:
    print(f"{texts=}")
    print(f"{labels=}")
    
    exit(0)
    


if __name__ == "__main__":
  main()