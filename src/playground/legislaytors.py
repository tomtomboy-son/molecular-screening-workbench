# legislators.py
# %%
import pandas as pd

dtypes = {
    "first_name": "category",
    "gender": "category",
    "type": "category",
    "state": "category",
    "party": "category",
}

df = pd.read_csv(
    "legislators-historical.csv",
    dtype=dtypes, # type: ignore
    usecols=list(dtypes) + ["birthday", "last_name"],
    parse_dates=["birthday"]
)
 # %%
