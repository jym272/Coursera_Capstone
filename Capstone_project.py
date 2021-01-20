import pandas as pd
from geopy.geocoders import Nominatim
import folium
import numpy as np
from translate import Translator
from sklearn.cluster import KMeans



cultural_spaces = pd.read_csv("espacios-culturales.csv")
cultural_spaces.shape
cultural_spaces.columns
cultural_spaces.head()
cultural_spaces.shape
cultural_spaces["FUNCION_PR"].value_counts()
cultural_spaces["FUNCION_PR"].isna().sum()
cultural_spaces["SUBCATEGOR"].isna().sum()

cultural_spaces["SUBCATEGOR"].value_counts()

cultural_spaces.groupby('FUNCION_PR')['SUBCATEGOR'].value_counts()

cultural_spaces[cultural_spaces["FUNCION_PR"] == "LIBRERIA"]



cultural_spaces_clean = cultural_spaces[['FUNCION_PR',"ESTABLECIM",'LATITUD', 'LONGITUD', 'BARRIO']]
cultural_spaces_clean

for value,i in zip(cultural_spaces_clean["FUNCION_PR"],range(3200)):
    translation = dic_venue_translations[value]
    cultural_spaces_clean.at[i,"FUNCION_PR"] = translation


dic_venue_translations = {"BIBLIOTECA": "Library",
    "LIBRERIA":"Bookshop",
    "ESPACIO ESCENICO":"Theater Space",
    "ESPACIO DE EXHIBICION":"Exhibition Space",
    "CENTRO CULTURAL":"Cultural Center",
    "MONUMENTOS Y LUGARES HISTORICOS":"Monuments and Historical Sites",
    "BAR":"Bar",
    "ESPACIO DE FORMACION":"Comprehensive Training Space",
    "DISQUERIA":"Record Store",
    "CALESITA":"Carousel",
    "ESPACIO FERIAL":"Carnival",
    "SALA DE CINE":"Cinema"}

types_cultural_spaces=cultural_spaces_clean["FUNCION_PR"].value_counts().to_frame()
types_cultural_spaces.reset_index(inplace=True)
types_cultural_spaces.rename(columns={'index':'TYPE',"FUNCION_PR":"number_cultural_spaces"},inplace=True)
types_cultural_spaces
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="darkgrid")
# %%
f, ax = plt.subplots(figsize=(6, 6))
sns.barplot(x="number_cultural_spaces",y="TYPE",data=types_cultural_spaces,color="b")
# legend
# ax.legend(ncol=2, loc="lower right", frameon=True)
ax.set(xlim=(0, 650), ylabel="",
       xlabel="Number of cultural spaces")
sns.despine(left=True, bottom=True)
# %%


cultural_spaces_clean.groupby('BARRIO')['FUNCION_PR'].count()

venues_each_neighborhood = cultural_spaces_clean.groupby('BARRIO')['FUNCION_PR'].value_counts()
venues_each_neighborhood.head(20)
venues_each_neighborhood["PALERMO"]
venues_each_neighborhood["VILLA RIACHUELO"]
# one hot encoding
caba_onehot = pd.get_dummies(cultural_spaces_clean[cultural_spaces_clean[["FUNCION_PR"]] != "DummySTR"]["FUNCION_PR"], prefix="", prefix_sep="")
# add neighborhood column back to dataframe
caba_onehot.insert(loc=0, column="Neighborhood", value=cultural_spaces_clean[['BARRIO']].values)
caba_onehot.shape
caba_onehot.head()

caba_grouped = caba_onehot.groupby('Neighborhood').mean().reset_index()
caba_grouped.describe()
caba_grouped

num_top_venues = 5

for hood in caba_grouped['Neighborhood']:
    print("----"+hood+"----")
    temp = caba_grouped[caba_grouped['Neighborhood'] == hood].T.reset_index()
    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))
    print('\n')

def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)

    return row_categories_sorted.index.values[0:num_top_venues]

num_top_venues = 5

indicators = ['st', 'nd', 'rd']
# create columns according to number of top venues
columns = ['Neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighborhood'] = caba_grouped['Neighborhood']

for ind in np.arange(caba_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(caba_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head(10)
neighborhoods_venues_sorted.describe()
caba_grouped



from sklearn.preprocessing import StandardScaler

caba_grouped_clustering = caba_grouped.drop('Neighborhood', 1)
X = caba_grouped_clustering.values
X = np.nan_to_num(X)
cluster_dataset = StandardScaler().fit_transform(X)
cluster_dataset


# set number of clusters
kclusters = 3
# run k-means clustering
kmeans = KMeans(init="k-means++",n_clusters=kclusters, random_state=0).fit(cluster_dataset)
# check cluster labels generated for each row in the dataframe
kmeans.labels_


import copy
# add clustering labels
neighborhoods_venues_clusters= copy.deepcopy(neighborhoods_venues_sorted)
neighborhoods_venues_clusters.insert(0, 'Cluster Labels', kmeans.labels_)

neighborhoods_venues_clusters
