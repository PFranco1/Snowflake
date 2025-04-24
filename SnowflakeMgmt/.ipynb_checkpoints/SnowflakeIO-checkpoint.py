# fastmatcher/io_utils.py
import pandas as pd
from typing import Iterator, List
import re

def csv_to_hdf5(
    csv_path: str,
    h5_path: str,
    key: str,
    chunk_size: int = 500_000,
    complib: str = 'blosc',
    complevel: int = 9,
    downcast_nums: bool = True,
    encoding: str = 'utf-8',
    errors: str = 'replace'
) -> None:
    """
    Read a CSV via a Python-level file handle (so bad bytes get replaced),
    then sanitize & append to a fixed-schema HDF5 table.
    """
    with pd.HDFStore(h5_path, mode='a', complevel=complevel, complib=complib) as store:
        # Remove old table (fresh start)
        if key in store:
            store.remove(key)

        template_cols = None

        # Open for *text* reading with replacement of any invalid bytes
        with open(csv_path, 'r', encoding=encoding, errors=errors) as fh:
            reader = pd.read_csv(fh, chunksize=chunk_size)

            for chunk in reader:
                # 1) reset index so pandas doesn’t introduce Level_0/Level_1
                chunk = chunk.reset_index(drop=True)

                # 2) downcast numeric columns
                if downcast_nums:
                    ints   = chunk.select_dtypes('integer').columns
                    floats = chunk.select_dtypes('floating').columns
                    for c in ints:
                        chunk[c] = pd.to_numeric(chunk[c], downcast='integer')
                    for c in floats:
                        chunk[c] = pd.to_numeric(chunk[c], downcast='float')

                # 3) sanitize column names
                clean = (
                    chunk.columns
                         .str.strip()
                         .str.replace(' ', '_', regex=False)
                         .str.replace(r'[^0-9A-Za-z_]', '', regex=True)
                )
                chunk.columns = clean

                # 4) fix template on first chunk, enforce thereafter
                if template_cols is None:
                    template_cols = list(chunk.columns)
                else:
                    chunk = chunk.reindex(columns=template_cols)

                # 5) append without the index
                store.append(key, chunk, data_columns=True, index=False)



def iter_tables(h5_path: str, prefix: str = '/') -> Iterator[str]:
    """
    Yield all table-keys in the HDF5 store that start with `prefix`.
    """
    with pd.HDFStore(h5_path, mode='r') as store:
        for key in store.keys():
            if key.startswith(prefix):
                yield key.strip('/')

def load_table(
    h5_path: str,
    key: str,
    columns: List[str] = None,
    where: str = None
) -> pd.DataFrame:
    """
    Load only the needed columns and/or filtered rows.
    """
    return pd.read_hdf(h5_path, key, columns=columns, where=where)
