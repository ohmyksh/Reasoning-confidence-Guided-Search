import os
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import pandas as pd
import argparse

class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

def encode_and_save_to_file(model_name_or_path, passage_file, encode_file_path, batch_size=16):
    df = pd.read_csv(passage_file, delimiter='\t')
    os.makedirs(encode_file_path, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModel.from_pretrained(model_name_or_path)
    model = torch.nn.DataParallel(model)
    model = model.cuda()

    documents = df['text'].tolist()
    dataset = TextDataset(documents)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for batch_idx, batch_documents in enumerate(tqdm(dataloader, desc="Encoding Documents")):
        inputs = tokenizer(batch_documents, return_tensors='pt', padding=True, truncation=True).to('cuda')
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)  # 예시: 평균 풀링

        for i, embedding in enumerate(embeddings):
            file_idx = batch_idx * batch_size + i
            file_path = os.path.join(encode_file_path, f"{file_idx // 1000}_{file_idx % 1000}.pt")
            torch.save(embedding.cpu(), file_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--passage_file", type=str, required=True)
    parser.add_argument("--encode_file_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    encode_and_save_to_file(
        model_name_or_path=args.model_name_or_path,
        passage_file=args.passage_file,
        encode_file_path=args.encode_file_path,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    main()
