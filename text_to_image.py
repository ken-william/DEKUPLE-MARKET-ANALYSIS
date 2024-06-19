# import pandas as pd
# from transformers import AutoTokenizer, AutoModel
# import torch
# import torch.nn.functional as F
# import re

# def load_data():
#     df = pd.read_csv("./dataset/image_annotations.csv")
#     category_dico = {
#         "Ao": "shirt",
#         "Balo": "backpack",
#         "Ca": "water animals",
#         "CanCau": "wands",
#         "Cho": "dogs",
#         "Ghe": "chairs",
#         "Giay": "shoes",
#         "Giuong": "beds",
#         "Heo": "pigs",
#         "Khac": "divers",
#         "Khoi": "cubes",
#         "Kinh": "glasses",
#         "Leu": "triangles",
#         "Meo": "cats",
#         "Non": "hats",
#         "Quan": "pants",
#         "Tho": "bunnies",
#         "Toc": "hair cuts",
#         "Trung": "eggs",
#         "Vay": "skirts",
#         "Xe": "cars"
#     }
#     df["category"] = df["Image Path"].apply(lambda x: category_dico[x.replace("train_cleaned\\", "").split("\\")[0]])
#     df = df.drop_duplicates("Caption").reset_index(drop=True)
#     return df

# def clean_caption(df):
#     df["Caption"] = df["Caption"].apply(lambda x: re.sub(r'\s* that is on the \w+ background', '', x))
#     df["Caption"] = df["Caption"].apply(lambda x: re.sub(r'\s* with a \w+ background', '', x))
#     df["Caption"] = df["Caption"].apply(lambda x: re.sub(r'\s* with \w+ background', '', x))
#     df["Caption"] = df["Caption"].apply(lambda x: re.sub(r'\s* on a \w+ background', '', x))
#     df["Caption"] = df["Caption"].apply(lambda x: re.sub(r'\s* on \w+ background', '', x))
#     df["Caption"] = df["Caption"].apply(lambda x: re.sub(r'\s* with a \w+ and \w+ background', '', x))
#     df["Caption"] = df["Caption"].apply(lambda x: re.sub(r'\s* with a \w+ and \w+ background', '', x))
#     df["Caption"] = df["Caption"].apply(lambda x: re.sub(r'\s* and a \w+ background', '', x))
#     df["Caption"] = df["Caption"].apply(lambda x: re.sub(r'\s* with clouds in the background', '', x))
#     df["Caption"] = df["Caption"].apply(lambda x: re.sub(r'\s* that is on a square \w+ background', '', x))
#     df["Caption"] = df["Caption"].apply(lambda x: re.sub(r'\s* on it against a \w+ background', '', x))
#     return df

# tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
# model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')

# def mean_pooling(model_output, attention_mask):
#     token_embeddings = model_output[0]
#     input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
#     return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# def get_score_model(sentences):
#     encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
#     with torch.no_grad():
#         model_output = model(**encoded_input)
#     sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
#     sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
#     cosine_similarity = torch.nn.functional.cosine_similarity(sentence_embeddings[0].unsqueeze(0), sentence_embeddings[1].unsqueeze(0))
#     return cosine_similarity.item()

# def recommend_images_based_on_text(prompt, num_images=5):
#     df = load_data()
#     df = clean_caption(df)
#     df["score"] = df["Caption"].apply(lambda caption: get_score_model([prompt, caption]))
#     df_final = df.sort_values("score", ascending=False).head(num_images)
#     return df_final



import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import re
import os

def load_data():
    df = pd.read_csv("./dataset/image_annotations.csv")
    category_dico = {
        "Ao": "shirt",
        "Balo": "backpack",
        "Ca": "water animals",
        "CanCau": "wands",
        "Cho": "dogs",
        "Ghe": "chairs",
        "Giay": "shoes",
        "Giuong": "beds",
        "Heo": "pigs",
        "Khac": "divers",
        "Khoi": "cubes",
        "Kinh": "glasses",
        "Leu": "triangles",
        "Meo": "cats",
        "Non": "hats",
        "Quan": "pants",
        "Tho": "bunnies",
        "Toc": "hair cuts",
        "Trung": "eggs",
        "Vay": "skirts",
        "Xe": "cars"
    }
    df["category"] = df["Image Path"].apply(lambda x: category_dico[x.replace("train_cleaned\\", "").split("\\")[0]])
    df = df.drop_duplicates("Caption").reset_index(drop=True)
    return df

def clean_caption(df):
    df["Caption"] = df["Caption"].apply(lambda x: re.sub(r'\s* that is on the \w+ background', '', x))
    df["Caption"] = df["Caption"].apply(lambda x: re.sub(r'\s* with a \w+ background', '', x))
    df["Caption"] = df["Caption"].apply(lambda x: re.sub(r'\s* with \w+ background', '', x))
    df["Caption"] = df["Caption"].apply(lambda x: re.sub(r'\s* on a \w+ background', '', x))
    df["Caption"] = df["Caption"].apply(lambda x: re.sub(r'\s* on \w+ background', '', x))
    df["Caption"] = df["Caption"].apply(lambda x: re.sub(r'\s* with a \w+ and \w+ background', '', x))
    df["Caption"] = df["Caption"].apply(lambda x: re.sub(r'\s* with a \w+ and \w+ background', '', x))
    df["Caption"] = df["Caption"].apply(lambda x: re.sub(r'\s* and a \w+ background', '', x))
    df["Caption"] = df["Caption"].apply(lambda x: re.sub(r'\s* with clouds in the background', '', x))
    df["Caption"] = df["Caption"].apply(lambda x: re.sub(r'\s* that is on a square \w+ background', '', x))
    df["Caption"] = df["Caption"].apply(lambda x: re.sub(r'\s* on it against a \w+ background', '', x))
    return df

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_score_model(sentences):
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    cosine_similarity = torch.nn.functional.cosine_similarity(sentence_embeddings[0].unsqueeze(0), sentence_embeddings[1].unsqueeze(0))
    return cosine_similarity.item()

def recommend_images_based_on_text(prompt, category, num_images=5):
    df = load_data()
    df = df[df['category'] == category]  # Filtrer par catégorie
    df = clean_caption(df)
    df["score"] = df["Caption"].apply(lambda caption: get_score_model([prompt, caption]))
    df_final = df.sort_values("score", ascending=False).head(num_images)
    return df_final
