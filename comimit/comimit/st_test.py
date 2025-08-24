"""
# My first app
Here's our first attempt at using data to create a table:
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import numpy as np
import pandas as pd
import json
import faiss
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import torch
st.set_page_config(layout='wide')
def paginator(label, items, items_per_page=10, on_sidebar=True):
    """Lets the user paginate a set of items.
    Parameters
    ----------
    label : str
        The label to display over the pagination widget.
    items : Iterator[Any]
        The items to display in the paginator.
    items_per_page: int
        The number of items to display per page.
    on_sidebar: bool
        Whether to display the paginator widget on the sidebar.
        
    Returns
    -------
    Iterator[Tuple[int, Any]]
        An iterator over *only the items on that page*, including
        the item's index.
    Example
    -------
    """

    # Figure out where to display the paginator
    if on_sidebar:
        location = st.sidebar.empty()
    else:
        location = st.empty()

    # Display a pagination selectbox in the specified location.
    items = list(items)
    n_pages = len(items)
    n_pages = (len(items) - 1) // items_per_page + 1
    page_format_func = lambda i: "Page %s" % i
    page_number = location.selectbox(label, range(n_pages), format_func=page_format_func)

    # Iterate over the items in the page to let the user display them.
    min_index = page_number * items_per_page
    max_index = min_index + items_per_page
    out_iter = iter([(" ".join(items[i].split("/")[1:]),items[i]) for i in range(len(items))])
    import itertools
    return itertools.islice(out_iter, min_index, max_index)

@st.cache_resource
def load_model(name = "sentence-transformers/clip-ViT-B-32"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return SentenceTransformer(name).to(device)

@st.cache_resource
def load_indexer(path = "/mnt/d/AIgov/new/frames_L14_FLATindexer.index"):
    return faiss.read_index(path)

@st.cache_data
def load_data(path = "/mnt/d/AIgov/new/map_org_all.parquet"):
    return pd.read_parquet(path)

@st.cache_data
def search_frame(text_query):
    text_emb = model.encode([text_query])
    input = text_emb
    faiss.normalize_L2(input)
    D,I = pq_frames.search(input,10000)
    return df_frames.iloc[I[0]].reset_index()

with st.sidebar:
    name = st.text_area("Query names",placeholder = "queries-p1-1")
    first = st.text_area("Top 1",placeholder = "L02_V016,14200")
option = st.selectbox(
    'Choose indexer',
    ('/mnt/d/AIgov/new/frames_L14_FLATindexer.index', '/mnt/d/AIgov/new/frames_L14_HNSWindexer.index'))

model = load_model()
pq_frames = load_indexer(option)
if st.button("Reload indexer"):
    del pq_frames
    pq_frames = load_indexer(option)
df_frames = load_data()
txt = st.text_area(
    "Query: ",
    "A fucking yellow plate toyota"
    )
st.write(txt)
df_out = search_frame(txt)
st.dataframe(df_out)

HOME = "compressed_2" # IMAGE FOLDER

list_img = [os.path.join(HOME,*df_out.iloc[pos][["Names",'frame name']].values.tolist()) for pos in range(0,500,1)] # SUA LAI LUON

image_iterator = paginator("result", list_img,items_per_page=100, on_sidebar=False)
indices_on_page, images_on_page = map(list, zip(*image_iterator))
st.image(images_on_page,width = 360, use_column_width="never", caption=indices_on_page)

with st.sidebar:
    if st.button("Confirm selection"):
        first_row = pd.Series({"Names": first.split(",")[0],
                            "frame idx": first.split(",")[-1]})
        df_out = pd.concat([first_row.to_frame().T, df_out])
        st.download_button(
            label="Download queries result",
            data=df_out.to_csv(),
            file_name=name,
            mime='text/csv',
        )