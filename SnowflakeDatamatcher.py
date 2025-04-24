# SnowflakeDatamatcher.py

import os
import re
import pandas as pd
from datasketch import MinHash, MinHashLSH
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from scipy.sparse import hstack

from SnowflakeMgmt.SnowflakeIO import csv_to_hdf5, iter_tables, load_table

class DataMatcher:
    def __init__(
        self,
        target_csv: str,
        raw_csvs: list[str],
        h5_store: str = 'data_store.h5',
        encoding: str = 'utf-8',
        errors: str = 'replace',
        debug: bool = False,
        header_threshold: float = 0.5,    # lowered for more permissive fuzzy
        data_threshold: float = 0.2,
        hdr_perm: int = 64,
        data_perm: int = 128
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

    def ingest(self):
        if os.path.exists(self.h5_store):
            os.remove(self.h5_store)
        csv_to_hdf5(self.target_csv, self.h5_store,
                    key='golden', encoding=self.encoding, errors=self.errors)
        for path in self.raw_csvs:
            clean = re.sub(r'[^0-9A-Za-z_]', '',
                           os.path.splitext(os.path.basename(path))[0]
                           .replace(' ', '_'))
            key = f"data_{clean}"
            csv_to_hdf5(path, self.h5_store, key=key,
                        encoding=self.encoding, errors=self.errors)

    def verify_store(self):
        with pd.HDFStore(self.h5_store, 'r') as store:
            print("HDF5 keys:", store.keys())

    @staticmethod
    def normalize_header(name: str) -> str:
        s = name.lower().replace('_','').replace(' ','')
        return re.sub(r'[^0-9a-z0-9]', '', s)

    @staticmethod
    def token_jaccard(a: str, b: str) -> float:
        ta = set(re.findall(r"\w+", a.lower()))
        tb = set(re.findall(r"\w+", b.lower()))
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / len(ta | tb)

    @staticmethod
    def column_jaccard(s1: pd.Series, s2: pd.Series) -> float:
        u1 = set(s1.dropna().astype(str).unique())
        u2 = set(s2.dropna().astype(str).unique())
        if not u1 or not u2:
            return 0.0
        return len(u1 & u2) / len(u1 | u2)

    def fuzzy_header_lsh(self, raw_cols: list[str]) -> MinHashLSH:
        lsh = MinHashLSH(threshold=self.header_threshold,
                         num_perm=self.hdr_perm)
        for c in raw_cols:
            m = MinHash(num_perm=self.hdr_perm)
            for token in re.findall(r"\w+", c.lower()):
                m.update(token.encode('utf8'))
            lsh.insert(c, m)
        return lsh

    def cluster(
        self,
        n_clusters: int = 9,
        batch_size: int = 4096,
        text_features: int = 5000
    ) -> pd.DataFrame:
        """Cluster the golden table rows and return it with a 'cluster' column."""
        df = load_table(self.h5_store, 'golden')
        num_cols = df.select_dtypes(include='number').columns.tolist()
        txt_cols = df.select_dtypes(include='object').columns.tolist()

        Xn = None
        if num_cols:
            Xn = StandardScaler().fit_transform(df[num_cols].fillna(0))

        Xt = None
        if txt_cols:
            combined = df[txt_cols].fillna('').agg(' '.join, axis=1)
            Xt = TfidfVectorizer(max_features=text_features).fit_transform(combined)

        # explicit combination logic
        if Xn is not None and Xt is not None:
            X = hstack([Xn, Xt])
        elif Xn is not None:
            X = Xn
        elif Xt is not None:
            X = Xt
        else:
            raise RuntimeError("No numeric or text features to cluster on")

        km = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=batch_size,
            random_state=42,
            n_init=10
        )
        labels = km.fit_predict(X)

        out = df.copy()
        out['cluster'] = labels
        return out


    def match(self) -> pd.DataFrame:
        # rebuild store if missing
        if not os.path.exists(self.h5_store):
            self.ingest()

        # load golden
        with pd.HDFStore(self.h5_store, 'r') as store:
            golden = store['golden']
        gold_cols = list(golden.columns)

        summary = []
        # iterate raw tables
        for tbl in iter_tables(self.h5_store, prefix='/'):
            name = tbl.strip('/')
            if not name.startswith('data_'):
                continue
            with pd.HDFStore(self.h5_store, 'r') as store:
                raw = store[tbl]
            raw_cols = list(raw.columns)

            # build LSH index for raw headers
            hdr_lsh = self.fuzzy_header_lsh(raw_cols)

            for gcol in gold_cols:
                hdr_exact = [c for c in raw_cols if c == gcol]

                # 1) fuzzy-header via LSH
                hdr_fuzzy = []
                if not hdr_exact:
                    m = MinHash(num_perm=self.hdr_perm)
                    for token in re.findall(r"\w+", gcol.lower()):
                        m.update(token.encode('utf8'))
                    hdr_fuzzy = hdr_lsh.query(m)

                # 2) fallback substring
                if not hdr_exact and not hdr_fuzzy:
                    ng = self.normalize_header(gcol)
                    hdr_fuzzy = [
                        c for c in raw_cols
                        if ng in self.normalize_header(c)
                        or self.normalize_header(c) in ng
                    ]

                # 3) fallback token‐Jaccard
                if not hdr_exact and not hdr_fuzzy:
                    for c in raw_cols:
                        j = self.token_jaccard(gcol, c)
                        if j >= self.header_threshold:
                            hdr_fuzzy.append(c)

                # exact-data
                data_exact = []
                for rc in raw_cols:
                    a = golden[gcol].fillna('').astype(str).reset_index(drop=True)
                    b = raw[rc].fillna('').astype(str).reset_index(drop=True)
                    if a.equals(b):
                        data_exact.append(rc)

                # fuzzy-data
                data_fuzzy = []
                if not data_exact:
                    for rc in raw_cols:
                        j = self.column_jaccard(golden[gcol], raw[rc])
                        if j >= self.data_threshold:
                            data_fuzzy.append((rc, round(j,2)))

                if hdr_exact or hdr_fuzzy or data_exact or data_fuzzy:
                    summary.append({
                        'raw_table': name,
                        'golden_col': gcol,
                        'hdr_exact': hdr_exact,
                        'hdr_fuzzy': hdr_fuzzy,
                        'data_exact': data_exact,
                        'data_fuzzy': data_fuzzy
                    })

        return pd.DataFrame(summary)

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        self.ingest()
        self.verify_store()
        summary_df  = self.match()
        clusters_df = self.cluster()
        return summary_df, clusters_df

def match_files(target: str, raws: list[str], debug: bool = False):
    dm = DataMatcher(target, raws, debug=debug)
    return dm.run()
