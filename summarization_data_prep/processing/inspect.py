from pyarrow.parquet import ParquetFile
import pyarrow as pa
import pandas as pd

import glob
for l in glob.glob('*.parquet'):
    pf = ParquetFile(l)
    first_ten_rows = next(pf.iter_batches(batch_size = 10))
    df = pa.Table.from_batches([first_ten_rows]).to_pandas()

    for i, row in df.iterrows():
        print(row['text'][:20])
        print('-----------\n\n')
        print(row['summary'][:20])
        print('----------\nPrompt\n')
        print(row['prompt'])
        print('-----------\n\n')