import os 
import sys
import pandas as pd
from typing import List
from beautifultable import BeautifulTable
import camelot
import ghostscript
import cv2
import camelot

print("IMPORT SUCCESSFUL")
file_path="./student_information_handbook_part_1.pdf"

# use camelot to parse tables   
def get_tables(path: str, pages: List[int]):    
    for page in pages:
        table_list = camelot.read_pdf(path, pages=str(page))
        if table_list.n>0:
            for tab in range(table_list.n):
                
                # Conversion of the the tables into the dataframes.
                table_df = table_list[tab].df 
                
                table_df = (
                    table_df.rename(columns=table_df.iloc[0])
                    .drop(table_df.index[0])
                    .reset_index(drop=True)
                )        
                     
                table_df = table_df.apply(lambda x: x.str.replace('\n',''))
                
                # Change column names to be valid as XML tags
                table_df.columns = [col.replace('\n', ' ').replace(' ', '') for col in table_df.columns]
                table_df.columns = [col.replace('(', '').replace(')', '') for col in table_df.columns]
    
    return table_df
# extract data table from page number
print("EXTRACTION SUCCESSFUL!")
df = get_tables(file_path, pages=[15])
print(df.head(100))