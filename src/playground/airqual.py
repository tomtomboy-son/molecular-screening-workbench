# airqual.py
# %%
import pandas as pd

df = pd.read_csv("airqual.csv",
        na_values=[-200],
        usecols=["Date", "Time", "CO(GT)", "T", "RH", "AH"]
        )

df["tstamp"] = pd.to_datetime(df.pop("Date") + ' ' + df.pop("Time"),
                    format="%m/%d/%y %H:%M:%S")

df.rename(columns={
        "CO(GT)": "co",
        "T": "temp_c",
        "RH": "rel_hum",
        "AH": "abs_hum",
        }, inplace=True  
        )

df.set_index("tstamp", inplace=True)
# %%
