# news.py
# %%
import pandas as pd

df = pd.read_csv("news.csv",
                sep="\t",
                header=None,
                index_col=0,
                names=["title", "url", "outlet", "category", "cluster","host","tstamp"],
                dtype={
                    "outlet":"category",
                    "category":"category",
                    "cluster": "category",
                    "host":"category",
                    },
                )

df["tstamp"] = pd.to_datetime(df["tstamp"], unit="ms")
# %%
