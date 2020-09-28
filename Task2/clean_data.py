#necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn import preprocessing
import warnings
warnings.filterwarnings('ignore')
#Loading data
df=pd.read_json('windows.json')
#Data treatment
df1= pd.DataFrame.transpose(df)
df1=df1.reset_index()
#price field
# We fill the nulls of price with the value of Price
df1['price'].fillna(value=df1['Price'],inplace=True)
# We remove the euro sign and change the data type to float
df1['price']=df1['price'].apply(lambda x:x.strip('€'))
df1['price']=df1['price'].str.replace(',', '').astype(float)
#substrate field
df1['Substrate'].fillna(value=df1['Substrate Material'],inplace=True)
df1['Substrate'].fillna(value=df1['Window Material'],inplace=True)
df1['Substrate'].fillna(value=df1['Material'],inplace=True)
df1[df1['category_description'].notnull()==True]['Substrate'].fillna(value='N-BK7',inplace=True)
#dimensions field
df1['Dimensions'].fillna(value=df1['Dimensions (mm)'],inplace=True)
df1['Dimensions'].fillna(value=df1['Dimensions (inches)'],inplace=True)
df1['Dimensions'].fillna(value=df1['Diameter (mm)'],inplace=True)
df1['Dimensions'].fillna(value=df1['Diameter φD'],inplace=True)
df1['Dimensions'].fillna(value=df1['Diameter'],inplace=True)
df1['Dimensions'].fillna(value=df1['D'],inplace=True)
df1['Dimensions'].fillna(value=df1['Minor Diameter'],inplace=True)
#Parallelism field
df1['Parallelism'].fillna(value=df1['Parallelism (arcmin)'],inplace=True)
df1['Parallelism'].fillna(value=df1['Parallelism (arcsec)'],inplace=True)
df1['Parallelism'].fillna(value=df1['Parallelism ･ Wedge angle'],inplace=True)
#Thickness field
df1['Thickness'].fillna(value=df1['Thickness (mm)'],inplace=True)
df1['Thickness'].fillna(value=df1['Thickness T'],inplace=True)
df1['Thickness'].fillna(value=df1['T, Thickness'],inplace=True)
df1['Thickness'].fillna(value=df1['Thickness'],inplace=True)
df1['Thickness'].fillna(value=df1['Thickness t'],inplace=True)
df1['Thickness'].fillna(value=df1['Edge Thickness ET (mm)'],inplace=True)
df1['Thickness'].fillna(value=df1['Maximum Edge Thickness (mm)'],inplace=True)
df1['Thickness'].fillna(value=df1['Center Thickness tc'],inplace=True)
df1['Thickness'].fillna(value=df1['Edge Thickness te'],inplace=True)
#Surface Quality field
df1['Surface Quality'].fillna(value=df1['Surface quality'],inplace=True)
df1['Surface Quality'].fillna(value=df1['Surface quality S1, S2'],inplace=True)
df1['Surface Quality'].fillna(value=df1['S1/S2 Surface Quality'],inplace=True)
df1['Surface Quality'].fillna(value=df1['Surface quality (scratch-dig)'],inplace=True)
#Wavelength field
df1['Wavelength Range'].fillna(value=df1['Wavelength Range (nm)'],inplace=True)
df1['Wavelength Range'].fillna(value=df1['Wavelength range'],inplace=True)
df1['Wavelength Range'].fillna(value=df1['Wavelength Range of AR Coating'],inplace=True)
#new dataframe df_clean
df_clean=df1[['index','availability','breadcrumbs','category','code_supplier','description_supplier','link','list_title',
 'price','supplier','technical_document_links','Substrate','Dimensions','Parallelism','Thickness','Surface Quality',
 'Wavelength Range']]
 #breadcrumbs field split
df_clean.drop(df_clean[df_clean['breadcrumbs'].apply(lambda x:len(x.split('>')))==4].index,inplace=True)
df_clean['bc1']=df_clean['breadcrumbs'].apply(lambda x:x.split('>')[0])
df_clean['bc2']=df_clean['breadcrumbs'].apply(lambda x:x.split('>')[1])
df_clean['bc3']=df_clean['breadcrumbs'].apply(lambda x:x.split('>')[2])
df_clean['bc4']=df_clean['breadcrumbs'].apply(lambda x:x.split('>')[3])
df_clean['bc5']=df_clean['breadcrumbs'].apply(lambda x:x.split('>')[4])
#Treatment Dimensions field
df_clean['clean_d']=df_clean['Dimensions']
df_clean['clean_d'].fillna(value='unknown',inplace=True)
df_clean['clean_d'].replace('\s','',regex=True,inplace=True)
df_clean['clean_d']=df_clean['clean_d'].str.replace('φ','')
df_clean['clean_d']=df_clean['clean_d'].str.replace('mm','')
df_clean['clean_d']=df_clean['clean_d'].str.replace('Ø1"','25.4')
df_clean['clean_d']=df_clean['clean_d'].str.replace('1"','25.4')
df_clean['clean_d']=df_clean['clean_d'].str.replace('"','')
df_clean['clean_d'].replace('(^\d+\.*\d*)(x)(\d+\.*\d*)(\W\d+\.\d+\W*\d*\.\d*)',r'\1\2\3',regex=True,inplace=True)
df_clean['clean_d'].replace('(^\d+\.*\d*)(x)(\d+\.*\d*)(\W\d+\.\d+\W*\d)',r'\1\2\3',regex=True,inplace=True)
df_clean['clean_d'].replace('(^\d+\.*\d*)(x)(\d+\.*\d*)(\W\d+\.\d+)',r'\1\2\3',regex=True,inplace=True)
df_clean['clean_d'].replace('(^\d+\.*\d*)(x)(\d+\.*\d*)([a-z]\d)',r'\1\2\3',regex=True,inplace=True)
df_clean['clean_d'].replace('(\d\/\d\()(\d+\.\d)(\))',r'\2',regex=True,inplace=True)
df_clean['clean_d'].replace('(\d\.*\d*\()(\d+\.\d)(\))',r'\2',regex=True,inplace=True)
df_clean['clean_d'].replace('(^\d+\.*\d*)(\W\d+\.*\d*\W*\d*\.\d+)',r'\1',regex=True,inplace=True)
df_clean['clean_d'].replace('(^\d+\.\d*)(\W\d+)',r'\1',regex=True,inplace=True)
df_clean['clean_d']=df_clean['clean_d'].str.replace(',','.')
df_clean['Dimensions']=df_clean['clean_d']
#isrectangular field
df_clean['isrectangular']=df_clean['Dimensions'].str.contains('(^\d+\.*\d*)(x)(\d+\.*\d*)',regex=True)
#height field
df_clean['height']=df_clean['Dimensions']
df_clean['height'].replace('(^\d+\.*\d*)(x)(\d+\.*\d*)',r'\1',regex=True,inplace=True)
df_clean.drop(df_clean[df_clean['height']=='unknown'].index,inplace=True)
df_clean['height']=df_clean['height'].astype(float)
#height field
df_clean['width']=df_clean['Dimensions']
df_clean['width'].replace('(^\d+\.*\d*)(x)(\d+\.*\d*)',r'\3',regex=True,inplace=True)
df_clean['width']=df_clean['width'].astype(float)
#Treatment Thickness
df_clean['Thickness'].replace('\s','',regex=True,inplace=True)
df_clean['Thickness']=df_clean['Thickness'].str.replace('mm','')
df_clean['Thickness']=df_clean['Thickness'].str.replace('(nominal)','')
df_clean['Thickness'].replace('(^\d+\.*\d*)(\W\d+\.*\d*\W*\d*\.\d+)',r'\1',regex=True,inplace=True)
df_clean['Thickness'].replace('(^\d+\.*\d*)(\(\))',r'\1',regex=True,inplace=True)
#Treatment availability
df_clean['availability']=df_clean['availability'].str.replace('3-5 Days','3-5 days')
df_clean['availability']=df_clean['availability'].str.replace('Contact Us','Request')
df_clean['availability']=df_clean['availability'].str.replace('Lead Time','Request')
#fill/dropna
df_clean['Substrate'].fillna(value='unknown',inplace=True)
df_clean['Parallelism'].fillna(value='unknown',inplace=True)
df_clean['Thickness'].fillna(value='unknown',inplace=True)
df_clean['Surface Quality'].fillna(value='unknown',inplace=True)
df_clean['Wavelength Range'].fillna(value='unknown',inplace=True)
df_clean.drop(df_clean[df_clean['Thickness']=='unknown'].index,inplace=True)
df_clean['Thickness']=df_clean['Thickness'].astype(float)
#create id_product field
LE = preprocessing.LabelEncoder()
df_clean['id_product'] = LE.fit_transform(df_clean['index'])
#Final Data Frame df_doc
df_doc=df_clean[['id_product','index','category','Substrate','Dimensions','isrectangular','height','width','availability','price','code_supplier','supplier','description_supplier','Thickness','bc2', 'bc3',
       'bc4','list_title','Parallelism','Surface Quality', 'Wavelength Range','link','technical_document_links']]
#renaming fields
df_doc.rename(columns={'index':'product','Substrate':'substrate','Dimensions':'dimensions','Thickness':'thickness',
                      'Parallelism':'parallelism','Surface Quality':'surface_quality','Wavelength Range':'wavelength_range',
                      'technical_document_links':'tech_doc_links'}, inplace=True)
#the new file
df_doc.to_json("windows_new.json")
