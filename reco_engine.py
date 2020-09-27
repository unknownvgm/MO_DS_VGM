#the libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
import warnings
warnings.filterwarnings('ignore')
#Loading data
df_doc=pd.read_json("windows_new.json")
df_use=df_doc[['id_product','product','category', 'substrate','dimensions','isrectangular', 'height',
 'width', 'availability', 'price','code_supplier', 'supplier','thickness', 'bc2','bc3', 'bc4', 'list_title',
 'parallelism', 'surface_quality', 'wavelength_range']]
#labels price and thickness field
bins_p = [0,50,100,150,200,300,500,700,800,1000,35000]
labels_p = ["<€50","€50-€100","€100-€150","€150-€200","€200-€300","€300-€500","€500-€700","€700-€800","€800-€1000",">€1000"]
df_use['price_bracket'] = pd.cut(df_use['price'], bins_p, labels=labels_p)
bins_t = [0,1,2,3,4,5,6,7,8,9,100]
labels_t = ["<mm1","mm1-mm2","mm2-mm3","mm3-mm4","mm4-mm5","mm5-mm6","mm6-mm7","mm7-mm8","mm8-mm9",">mm9"]
df_use['thickness_bracket'] = pd.cut(df_use['thickness'], bins_t, labels=labels_t)
#Data frame for the recommendation engine
df_train=df_use[['substrate','supplier','isrectangular','dimensions','availability','price_bracket','thickness_bracket', 'bc2', 'bc3', 'bc4', 'list_title']]
df_train["price_bracket"]=df_train["price_bracket"].astype(str)
df_train["thickness_bracket"]=df_train["thickness_bracket"].astype(str)
df_train["isrectangular"]=df_train["isrectangular"].astype(str)
#fill the possible nulls
features = ['substrate','supplier','isrectangular','dimensions','availability','price_bracket',
'thickness_bracket','bc2','bc3','bc4','list_title']
for feature in features:
    df_train[feature] = df_train[feature].fillna('')
# combined features field
df_train["combined_features"]=df_train['substrate']+" "+df_train["supplier"]+" "+df_train["isrectangular"]+" "+df_train["dimensions"]+" "+df_train["availability"]+" "+df_train["price_bracket"]+" "+df_train["thickness_bracket"]+" "+df_train["bc2"]+" "+df_train["bc3"]+" "+df_train["bc4"]+" "+df_train["list_title"]
df_train['combined_features']=df_train['combined_features'].astype(str)
# remove stopwords
stop_words = stopwords.words('english')
df_train['combined_features'] = df_train['combined_features'].str.lower().str.split()
df_train["features"]=df_train["combined_features"].apply(lambda x: [word for word in x if word not in stop_words])
df_train["features"]=df_train["features"].apply(lambda x: " ".join(x))
#Count Matrix
cv = CountVectorizer()
count_matrix = cv.fit_transform(df_train["features"])
cosine_sim = cosine_similarity(count_matrix)
#functions to get id or product
def get_product_from_index(index):
    return df_use[df_use['id_product'] == index]["product"].values[0]
def get_id_from_product(title):
    return df_use[df_use['product']== title]["id_product"].values[0]
#let's try recommendation
product_user_likes = 'product_#02-075'
product_index = get_id_from_product(product_user_likes)
similar_products =  list(enumerate(cosine_sim[product_index]))
sorted_similar_products = sorted(similar_products,key=lambda x:x[1],reverse=True)
i=0
for element in sorted_similar_products:
    print (get_product_from_index(element[0]),element[1])
    i=i+1
    if i>10:
        break
#top10 similar products Data frame
df_sim=pd.DataFrame(sorted_similar_products)
df_sim.rename(columns={0:'id_product',1:'similarity'}, inplace=True)
df_sim10=df_sim[1:11]
df_sim10['product']=df_sim10['id_product'].apply(lambda x:get_product_from_index(x))
df_sim10=df_sim10[['id_product','product','similarity']]
df_sim10.to_json('top10_recommendations.json')