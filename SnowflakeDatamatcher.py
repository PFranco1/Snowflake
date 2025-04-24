# ─────────────────────────────────────────────────────────────────────────────
# SnowflakeDatamatcher Worksheet
#
# Paste or import your SnowflakeMgmt.SnowflakeIO & .SnowflakeMatch here,
# then the matcher class, then this main() function.
# ─────────────────────────────────────────────────────────────────────────────

import re
import pandas as pd
from datasketch import MinHash, MinHashLSH
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from scipy.sparse import hstack

from snowflake.snowpark import Session

# If you’ve staged your helpers (SnowflakeMgmt) as a .zip,
# uncomment and adjust the following:
# IMPORTS = ('@my_code_stage/my_matcher.zip',)
# from SnowflakeMgmt.SnowflakeIO import csv_to_hdf5, iter_tables, load_table

# For inline worksheets, we just re‐use our load_table helper to pull
# directly from Snowpark into pandas:
def load_table(session: Session, table_name: str) -> pd.DataFrame:
    return session.table(table_name).to_pandas()

class SnowparkDataMatcher:
    def __init__(
        self,
        session: Session,
        golden_table: str,
        raw_tables: list[str],
        debug: bool = False,
        header_threshold: float = 0.6,
        data_threshold: float = 0.2,
        hdr_perm: int = 64,
        data_perm: int = 128
    ):
        self.session          = session
        self.golden_table     = golden_table
        self.raw_tables       = raw_tables
        self.debug            = debug
        self.header_threshold = header_threshold
        self.data_threshold   = data_threshold
        self.hdr_perm         = hdr_perm
        self.data_perm        = data_perm

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

    def _fuzzy_header_lsh(self, raw_cols: list[str]) -> MinHashLSH:
        lsh = MinHashLSH(threshold=self.header_threshold,
                         num_perm=self.hdr_perm)
        for c in raw_cols:
            m = MinHash(num_perm=self.hdr_perm)
            for tok in re.findall(r"\w+", c.lower()):
                m.update(tok.encode('utf8'))
            lsh.insert(c, m)
        return lsh

    def match(self) -> pd.DataFrame:
        # load golden once
        golden = load_table(self.session, self.golden_table)
        gold_cols = list(golden.columns)

        summary = []
        for tbl in self.raw_tables:
            raw = load_table(self.session, tbl)
            raw_cols = list(raw.columns)
            if self.debug:
                print(f"\n--- Matching {self.golden_table} vs {tbl} ---")
                print(" Raw cols:", raw_cols)

            hdr_lsh = self._fuzzy_header_lsh(raw_cols)
            for gcol in gold_cols:
                # exact header
                hdr_exact = [c for c in raw_cols if c == gcol]
                # fuzzy header via LSH
                hdr_fuzzy = []
                if not hdr_exact:
                    m = MinHash(num_perm=self.hdr_perm)
                    for tok in re.findall(r"\w+", gcol.lower()):
                        m.update(tok.encode('utf8'))
                    hdr_fuzzy = hdr_lsh.query(m)
                    # substring fallback
                    if not hdr_fuzzy:
                        ng = self.normalize_header(gcol)
                        hdr_fuzzy = [c for c in raw_cols
                                     if ng in self.normalize_header(c)]
                    # token-jaccard fallback
                    if not hdr_fuzzy:
                        for c in raw_cols:
                            j = self.token_jaccard(gcol, c)
                            if j >= self.header_threshold:
                                hdr_fuzzy.append(c)

                # exact data
                data_exact = []
                for rc in raw_cols:
                    a = golden[gcol].fillna('').astype(str).reset_index(drop=True)
                    b = raw[rc].fillna('').astype(str).reset_index(drop=True)
                    if a.equals(b):
                        data_exact.append(rc)

                # fuzzy data
                data_fuzzy = []
                if not data_exact:
                    for rc in raw_cols:
                        j = self.column_jaccard(golden[gcol], raw[rc])
                        if j >= self.data_threshold:
                            data_fuzzy.append((rc, round(j,2)))

                if hdr_exact or hdr_fuzzy or data_exact or data_fuzzy:
                    summary.append({
                        'raw_table': tbl,
                        'golden_col': gcol,
                        'hdr_exact': hdr_exact,
                        'hdr_fuzzy': hdr_fuzzy,
                        'data_exact': data_exact,
                        'data_fuzzy': data_fuzzy
                    })

        return pd.DataFrame(summary)

    def cluster(self) -> pd.DataFrame:
        df = load_table(self.session, self.golden_table)
        num_cols = df.select_dtypes(include='number').columns.tolist()
        txt_cols = df.select_dtypes(include='object').columns.tolist()

        Xn = (StandardScaler().fit_transform(df[num_cols].fillna(0))
              if num_cols else None)
        Xt = None
        if txt_cols:
            combined = df[txt_cols].fillna('').agg(' '.join, axis=1)
            Xt = TfidfVectorizer(max_features=5000).fit_transform(combined)

        if Xn is not None and Xt is not None:
            X = hstack([Xn, Xt])
        else:
            X = Xn or Xt

        km = MiniBatchKMeans(n_clusters=5,
                             batch_size=4096,
                             random_state=42,
                             n_init=10)
        labels = km.fit_predict(X)
        out = df.copy()
        out['cluster'] = labels
        return out

    def run(self) -> tuple[pd.DataFrame,pd.DataFrame]:
        """Run match + cluster, return (summary_df, clusters_df)."""
        summary_df  = self.match()
        clusters_df = self.cluster()
        return summary_df, clusters_df

# ─────────────────────────────────────────────────────────────────────────────
# Handler required by Python Worksheet
# ─────────────────────────────────────────────────────────────────────────────

def main(session: Session):
    """
    session: the Snowpark Session object injected by the Worksheet runtime.
    """
    matcher = SnowparkDataMatcher(
        session=session,
        golden_table="MY_DB.MY_SCHEMA.DIAMOND_CLIENTS_LIST",
        raw_tables=[
          "MY_DB.MY_SCHEMA.TEST1",
          "MY_DB.MY_SCHEMA.LST_ASSUMPTION_DASHBOARD",
          "MY_DB.MY_SCHEMA.PRODUCT_HIERARCHY_EXTERNAL"
        ],
        debug=False
    )
    summary_df, clusters_df = matcher.run()
    # Return a Snowpark DataFrame so it displays in the worksheet:
    return session.create_dataframe(summary_df)

# Now just press “Run” in your worksheet.
# You’ll see a table of summary_df as the output.
# If you also want clusters_df, you can display it manually:
#
# _, clusters = matcher.run()
# display(session.create_dataframe(clusters))
