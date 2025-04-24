# SnowflakeDatamatcher.py

import os, re
import pandas as pd
from datasketch import MinHash, MinHashLSH
from .SnowflakeCluster import *
from .SnowflakeIO import *

class DataMatcher:
    def __init__(
        self,
        target_csv,
        raw_csvs,
        h5_store='data_store.h5',
        encoding='utf-8',
        errors='replace',
        debug=False,
        header_threshold=0.01,
        data_threshold=0.01,
        hdr_perm=64,
        data_perm=128
    ):
        self.target_csv       = target_csv
        self.raw_csvs         = raw_csvs
        self.h5_store         = h5_store
        self.encoding         = encoding
        self.errors           = errors
        self.debug            = debug
        self.header_threshold = header_threshold
        self.data_threshold   = data_threshold
        self.hdr_perm         = hdr_perm
        self.data_perm        = data_perm

        self.header_matches = {}
        self.data_matches   = {}

    def ingest(self):
        if os.path.exists(self.h5_store):
            os.remove(self.h5_store)
        print(f"Ingesting {self.target_csv} → key 'golden'")
        csv_to_hdf5(self.target_csv, self.h5_store,
                    key='golden', encoding=self.encoding, errors=self.errors)
        for path in self.raw_csvs:
            clean = re.sub(r'[^0-9A-Za-z_]', '',
                           os.path.splitext(os.path.basename(path))[0]
                           .replace(' ', '_'))
            key = f"data_{clean}"
            print(f"Ingesting {path} → key '{key}'")
            csv_to_hdf5(path, self.h5_store,
                        key=key, encoding=self.encoding, errors=self.errors)

    def verify_store(self):
        store = pd.HDFStore(self.h5_store, 'r')
        print("\nFINAL HDF5 KEYS:", store.keys())
        store.close()

    def inspect_store(self):
        print("\n=== Inspecting HDF5 Store (first 3 rows each) ===")
        for tbl in iter_tables(self.h5_store, prefix=''):
            df = load_table(self.h5_store, tbl).head(3)
            print(f"\nTable: {tbl}\n Columns: {list(df.columns)}\n{df}")

    @staticmethod
    def normalize_header(name: str) -> str:
        s = name.lower().replace('_','').replace(' ','')
        return re.sub(r'[^0-9a-z]', '', s)

    @staticmethod
    def column_jaccard(s1: pd.Series, s2: pd.Series) -> float:
        u1 = set(s1.dropna().astype(str).unique())
        u2 = set(s2.dropna().astype(str).unique())
        if not u1 or not u2:
            return 0.0
        return len(u1 & u2) / len(u1 | u2)

    @staticmethod
    def _name_minhash(name: str, num_perm: int) -> MinHash:
        m = MinHash(num_perm=num_perm)
        s = re.sub(r'[_\s]+',' ', name.strip())
        toks = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])", s)
        for t in toks:
            m.update(t.lower().encode('utf8'))
        return m

    def match(self):
        print("\n>>> Entering match()")  # <<< this confirms match() is running
        golden = load_table(self.h5_store, 'golden')
        gold_cols = list(golden.columns)

        for table_key in iter_tables(self.h5_store, prefix='data_'):
            raw = load_table(self.h5_store, table_key)
            raw_cols = list(raw.columns)

            print(f"\n--- Matching against {table_key} ---")
            print(" Golden cols:", gold_cols)
            print(" Raw    cols:", raw_cols)

            # build fuzzy header LSH
            hdr_lsh = MinHashLSH(threshold=self.header_threshold,
                                 num_perm=self.hdr_perm)
            for c in raw_cols:
                hdr_lsh.insert(c, self._name_minhash(c, self.hdr_perm))

            # build fuzzy data LSH
            data_lsh = build_lsh_index(raw, raw_cols,
                                       threshold=self.data_threshold,
                                       num_perm=self.data_perm)

            for gcol in gold_cols:
                print(f"\nGolden column → '{gcol}'")

                # 1) HEADER exact normalized
                ng = self.normalize_header(gcol)
                exact_hdr = [c for c in raw_cols
                             if self.normalize_header(c) == ng]
                if exact_hdr:
                    print(" [HEADER] normalized exact →", exact_hdr)
                    self.header_matches[(table_key, gcol)] = exact_hdr
                else:
                    # 2) HEADER fuzzy
                    m1   = self._name_minhash(gcol, self.hdr_perm)
                    hits = hdr_lsh.query(m1)
                    if hits:
                        details = []
                        for h in hits:
                            j = m1.jaccard(self._name_minhash(h, self.hdr_perm))
                            details.append(f"{h}(J={j:.2f})")
                        print(" [HEADER] fuzzy →", details)
                        self.header_matches[(table_key, gcol)] = hits
                    else:
                        print(" [HEADER]  -- no match")

                # 3) DATA exact row‐equals
                exact_data = []
                for rcol in raw_cols:
                    a = golden[gcol].fillna('').astype(str).reset_index(drop=True)
                    b = raw[rcol].fillna('').astype(str).reset_index(drop=True)
                    if a.equals(b):
                        exact_data.append(rcol)
                if exact_data:
                    print(" [DATA] exact →", exact_data)
                    self.data_matches[(table_key, gcol)] = exact_data
                else:
                    # 4) DATA fuzzy
                    fuzz = []
                    for rcol in raw_cols:
                        score = self.column_jaccard(golden[gcol], raw[rcol])
                        if score >= self.data_threshold:
                            fuzz.append(f"{rcol}(J={score:.2f})")
                    if fuzz:
                        print(" [DATA] fuzzy →", fuzz)
                        self.data_matches[(table_key, gcol)] = [
                            f.split('(J=')[0] for f in fuzz
                        ]
                    else:
                        print(" [DATA]  -- no match")

    def run(self):
        print("\n>>> Starting DataMatcher.run()  debug =", self.debug)
        self.ingest()
        self.verify_store()
        if self.debug:
            self.inspect_store()
        print("\n>>> Now running match() …")
        self.match()
        print("\n>>> Finished match()")

def match_files(target, raws, debug=False):
    dm = DataMatcher(target, raws, debug=debug)
    dm.run()
