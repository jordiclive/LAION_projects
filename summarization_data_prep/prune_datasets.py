import pandas as pd
from pyarrow.parquet import ParquetFile
import pyarrow as pa
import re

def word_count(x):
    count = len(re.findall(r"\w+", x))
    return count


for path in ['df_withDescription.parquet', 'df_withoutDescription.parquet']:
    df = pd.read_parquet(path)

    df['text_len'] = 1.42 * df["full_text"].apply(word_count)
    df['summary_len'] = 1.42 * df["summary"].apply(word_count)

    df = df[(df['text_len'] < 750) & (df['summary_len'] < 151)]
    df['text_ratio'] = df['summary_len'] / df['text_len']
    df = df[df['text_ratio'] < 0.5]
    df.reset_index(inplace=True, drop=True)
    df = df.drop_duplicates(subset=['full_text', 'summary'])
    df.reset_index(inplace=True, drop=True)
    df.to_parquet(path)
#
df1 = pd.read_parquet('df_withDescription.parquet')
df2 = pd.read_parquet('df_withoutDescription.parquet')
df1 = df1[list(df2.columns)]
df = pd.concat([df1, df2])
df.reset_index(inplace=True, drop=True)
df = df.drop_duplicates(subset=['full_text', 'summary'])
df.reset_index(inplace=True, drop=True)
df = df.rename({'full_text':'text'}, axis=1)
df['contriever_cos'] = 0
df.to_parquet('wiki-summary.parquet')



# path = '/Users/jordanclive/Personal_git/LAION_projects/summarization_data_prep/df_withoutDescription.parquet'
# pf = ParquetFile(path)
# first_ten_rows = next(pf.iter_batches(batch_size = 1000))
# df = pa.Table.from_batches([first_ten_rows]).to_pandas()
#
# path = '/Users/jordanclive/Personal_git/LAION_projects/summarization_data_prep/df_withDescription.parquet'
#
# df = pd.read_parquet(path)
#
# df['text_len'] = 1.42 *df["full_text"].apply(word_count)
# df['summary_len'] = 1.42 *df["summary"].apply(word_count)
#
# df = df[(df['text_len'] < 750) & (df['summary_len'] < 151)]
# df['text_ratio'] = df['summary_len'] / df['text_len']
# df = df[df['text_ratio'] < 0.5]
# df.reset_index(inplace=True,drop=True)
# df = df.drop_duplicates(subset=['full_text', 'summary'])
# df.reset_index(inplace=True,drop=True)
# df.to_parquet('df_withDescription_updated.parquet')

df1 = pd.read_parquet('/Users/jordanclive/Personal_git/LAION_projects/summarization_data_prep/df_withDescription_updated.parquet')
df2 =pd.read_parquet('/Users/jordanclive/Personal_git/LAION_projects/summarization_data_prep/df_withoutDescription_updated.parquet')
df1 = df1[list(df2.columns)]
df = pd.concat([df1,df2])
df.reset_index(inplace=True,drop=True)
df = df.drop_duplicates(subset=['full_text', 'summary'])
df.reset_index(inplace=True,drop=True)
df.to_parquet('wiki-summary.parquet')
print(len(df))










## Billsum prune



# df = pa.Table.from_batches([first_ten_rows]).to_pandas()
# df = pd.read_parquet("/Users/jordanclive/Personal_git/LAION_projects/summarization_data_prep/scored_summarization_datasets/datasets/billsum_train_scored.snappy.parquet")
# print("Dataset Length:", len(df))
# print("\n\ntext Word Count")
# print_quantiles(df["text"].apply(word_count))
# print("\n\nSummary Word Count")
# print_quantiles(df["summary"].apply(word_count))
# # print('\n\ntext Token Count')
# # print_quantiles(df['text'].apply(token_count))
# # print('\n\nSummary Token Count')
# # print_quantiles(df['summary'].apply(token_count))
#
# print("\n\ntext token calc Count")
# print_quantiles(1.42 * df["text"].apply(word_count))
# print("\n\nSummary token calc Count")
# print_quantiles(1.42 * df["summary"].apply(word_count))
# print("\n\ntext Token Count")

df['text_len'] = 1.42 *df["text"].apply(word_count)
df['summary_len'] = 1.42 *df["summary"].apply(word_count)
df = df[(df['text_len'] < 750) & (df['summary_len'] < 151)]
df['text_ratio'] = df['summary_len'] / df['text_len']
df = df[df['text_ratio'] < 0.5]