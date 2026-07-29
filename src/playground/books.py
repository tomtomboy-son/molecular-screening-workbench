# %%
import pandas as pd

def read():
    return (pd.read_csv("data-sets/BL-Flickr-Images-Book.csv")
            .rename(columns= lambda header: header.lower().replace(" ", "_"))
            .rename(columns= {"identifier": "id"})
            .drop(columns=[
                'edition_statement', 
                'contributors',
                'corporate_author', 
                'corporate_contributors', 
                'former_owner',
                'engraver', 
                'issuance_type', 
                'flickr_url', 
                'shelfmarks'
                ]
            )
            .set_index("id")
            .assign(date_of_publication=clean_date_of_publication)
            .assign(place_of_publication=clean_place_of_publication)
        )

def clean_date_of_publication(books):
    return(pd.to_numeric(
        books.loc[:, "date_of_publication"].str
        .extract(r"(\d{4})", expand=False)
        .fillna(0),
        downcast="unsigned"
        )
    )

def clean_place_of_publication(books):
    return (books.loc[:, "place_of_publication"]
            .str.replace(r".*London.*", "London", regex=True)
            .str.replace(r"-", " ", regex=True)
            .str.replace(r".*Oxford.*", "Oxford", regex=True)
            .str.replace(r".*Plymouth.*", "Plymouth", regex=True)
    )

books = read()
 # %%
