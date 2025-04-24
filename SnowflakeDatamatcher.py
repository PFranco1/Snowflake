import re
import pandas as pd

from snowflake.snowpark import Session
from snowflake.snowpark.functions import col

from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from scipy.sparse import hstack

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_table(session: Session, table_name: str) -> pd.DataFrame:
    """Read an entire Snowflake table into pandas."""
    return session.table(table_name).to_pandas()

def normalize_header(name: str) -> str:
    s = name.lower().replace('_','').replace(' ','')
    return re.sub(r'[^0-9a-z0-9]', '', s)

def token_jaccard(a: str, b: str) -> float:
    ta = set(re.findall(r"\w+", a.lower()))
    tb = set(re.findall(r"\w+", b.lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)

def column_jaccard(s1: pd.Series, s2: pd.Series) -> float:
    u1 = set(s1.dropna().astype(str).unique())
    u2 = set(s2.dropna().astype(str).unique())
    if not u1 or not u2:
        return 0.0
    return len(u1 & u2) / len(u1 | u2)

# ─────────────────────────────────────────────────────────────────────────────
# Matcher
# ─────────────────────────────────────────────────────────────────────────────

class SnowparkDataMatcher:
    def __init__(
        self,
        session: Session,
        golden_table: str,
        raw_tables: list[str],
        header_threshold: float = 0.6,
        data_threshold: float   = 0.2,
        debug: bool             = False
    ):
        self.session          = session
        self.golden_table     = golden_table
        self.raw_tables       = raw_tables
        self.header_threshold = header_threshold
        self.data_threshold   = data_threshold
        self.debug            = debug

    def match(self) -> pd.DataFrame:
        # 1) Load golden
        golden = load_table(self.session, self.golden_table)
        gold_cols = list(golden.columns)

        summary = []
        # 2) Iterate through raw tables
        for tbl in self.raw_tables:
            raw = load_table(self.session, tbl)
            raw_cols = list(raw.columns)
            if self.debug:
                print(f"\nMatching against {tbl}")
                print(" Raw columns:", raw_cols)

            for gcol in gold_cols:
                # exact header
                hdr_exact = [c for c in raw_cols if c == gcol]

                # fuzzy header: token_jaccard
                hdr_fuzzy = []
                if not hdr_exact:
                    for c in raw_cols:
                        j = token_jaccard(gcol, c)
                        if j >= self.header_threshold:
                            hdr_fuzzy.append((c, round(j,2)))

                # exact data
                data_exact = []
                for c in raw_cols:
                    a = golden[gcol].fillna('').astype(str).reset_index(drop=True)
                    b = raw[c].fillna('').astype(str).reset_index(drop=True)
                    if a.equals(b):
                        data_exact.append(c)

                # fuzzy data: column_jaccard
                data_fuzzy = []
                if not data_exact:
                    for c in raw_cols:
                        j = column_jaccard(golden[gcol], raw[c])
                        if j >= self.data_threshold:
                            data_fuzzy.append((c, round(j,2)))

                # record only if any matches
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

    def cluster(
        self,
        n_clusters: int    = 5,
        text_features: int = 5000
    ) -> pd.DataFrame:
        # Load golden again
        df = load_table(self.session, self.golden_table)
        num_cols = df.select_dtypes(include='number').columns.tolist()
        txt_cols = df.select_dtypes(include='object').columns.tolist()

        # Numeric features
        Xn = (StandardScaler().fit_transform(df[num_cols].fillna(0))
              if num_cols else None)
        # Text features
        Xt = None
        if txt_cols:
            combined = df[txt_cols].fillna('').agg(' '.join, axis=1)
            Xt = TfidfVectorizer(max_features=text_features).fit_transform(combined)

        # Combine
        if Xn is not None and Xt is not None:
            X = hstack([Xn, Xt])
        elif Xn is not None:
            X = Xn
        else:
            X = Xt

        km = MiniBatchKMeans(n_clusters=n_clusters,
                             batch_size=4096,
                             random_state=42,
                             n_init=10)
        labels = km.fit_predict(X)

        out = df.copy()
        out['cluster'] = labels
        return out

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        return self.match(), self.cluster()

# ─────────────────────────────────────────────────────────────────────────────
# Required handler for a Python Worksheet
# ─────────────────────────────────────────────────────────────────────────────

def main(session: Session):
    """
    Snowflake will inject `session` here. We return a Snowpark DataFrame
    so it renders as a table in the worksheet.
    """
    matcher = SnowparkDataMatcher(
        session=session,
        golden_table="MY_DB.MY_SCHEMA.DIAMOND_CLIENTS_LIST",
        raw_tables=[
            "MY_DB.MY_SCHEMA.TEST1",
            "MY_DB.MY_SCHEMA.LST_ASSUMPTION_DASHBOARD",
            "MY_DB.MY_SCHEMA.PRODUCT_HIERARCHY_EXTERNAL"
        ],
        header_threshold=0.6,
        data_threshold=0.2,
        debug=False
    )

    summary_df, clusters_df = matcher.run()
    # display the summary
    return session.create_dataframe(summary_df)
