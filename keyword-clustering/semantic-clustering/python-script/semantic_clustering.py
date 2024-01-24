import sys
import os
import time
import pandas as pd
import questionary
import glob
from sentence_transformers import SentenceTransformer, util

# Model for computing sentence embeddings. We use one trained for similar questions detection
model = SentenceTransformer('paraphrase-MiniLM-L3-v2')  #1861  /  7.7
  
  
# use glob to get all the csv files 
# in the folder
path = os.getcwd()
csv_files = glob.glob(os.path.join(path, "*.csv"))
  
  
# loop over the list of csv files
for f in csv_files:
      
    # read the csv file
    df = pd.read_csv(f)
    
cols = df.columns.tolist()

choice = questionary.select("Please select your keyword column", choices=cols).ask()  # returns value of selection

# store the data
cluster_name_list = []
corpus_sentences_list = []
df_all = []

corpus_set = set(df[choice])
corpus_set_all = corpus_set

cluster = True

while cluster:

    corpus_sentences = list(corpus_set)
    check_len = len(corpus_sentences)

    corpus_embeddings = model.encode(corpus_sentences, batch_size=256, show_progress_bar=True, convert_to_tensor=True)
    clusters = util.community_detection(corpus_embeddings, min_community_size=2, threshold=0.75, init_max_size=len(corpus_embeddings))

    for keyword, cluster in enumerate(clusters):
        print("\nCluster {}, #{} Elements ".format(keyword + 1, len(cluster)))

        for sentence_id in cluster[0:]:
            print("\t", corpus_sentences[sentence_id])
            corpus_sentences_list.append(corpus_sentences[sentence_id])
            cluster_name_list.append("Cluster {}, #{} Elements ".format(keyword + 1, len(cluster)))

    df_new = pd.DataFrame(None)
    df_new['Cluster Name'] = cluster_name_list
    df_new["Keyword"] = corpus_sentences_list

    df_all.append(df_new)
    have = set(df_new["Keyword"])

    corpus_set = corpus_set_all - have
    remaining = len(corpus_set)

    if check_len == remaining:
        break

df_new = pd.concat(df_all)


df = df.merge(df_new.drop_duplicates('Keyword'), how='left', on="Keyword")

# ------------------------------ rename the clusters to the shortest keyword -------------------------------------------

df['Length'] = df['Keyword'].astype(str).map(len)
df = df.sort_values(by="Length", ascending=True)

df['Cluster Name'] = df.groupby('Cluster Name')['Keyword'].transform('first')
df.sort_values(['Cluster Name', "Keyword"], ascending=[True, True], inplace=True)

df['Cluster Name'] = df['Cluster Name'].fillna("zzz_no_cluster")

del df['Length']

col = df.pop("Keyword")
df.insert(0, col.name, col)

col = df.pop('Cluster Name')
df.insert(0, col.name, col)

df.sort_values(["Cluster Name", "Keyword"], ascending=[True, True], inplace=True)

newpath = path + 'output' 
if not os.path.exists(newpath):
    os.makedirs(newpath)

df.to_csv(path + newpath + "test.csv")