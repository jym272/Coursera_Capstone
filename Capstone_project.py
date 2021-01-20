import pandas as pd
from geopy.geocoders import Nominatim
import folium
import numpy as np
from sklearn.cluster import KMeans



cultural_spaces = pd.read_csv("espacios-culturales.csv")
cultural_spaces.shape
cultural_spaces.columns
cultural_spaces.head()
cultural_spaces["FUNCION_PR"].value_counts()
cultural_spaces["FUNCION_PR"].isna().sum()
cultural_spaces["SUBCATEGOR"].isna().sum()

cultural_spaces["SUBCATEGOR"].value_counts()

cultural_spaces[cultural_spaces.SUBCATEGOR == "UNIVERSITARIA Y/O CIENTIFICO TECNICA"].groupby("BARRIO")["BARRIO"].value_counts()


cultural_spaces.groupby('FUNCION_PR')['SUBCATEGOR'].value_counts()

cultural_spaces[cultural_spaces["FUNCION_PR"] == "LIBRERIA"]


cultural_spaces_clean = cultural_spaces[['FUNCION_PR',"ESTABLECIM",'LATITUD', 'LONGITUD', 'BARRIO']]
cultural_spaces_clean.rename(columns={'ESTABLECIM':'Name',"FUNCION_PR":"Cultural space"})
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


for value,i in zip(cultural_spaces_clean["FUNCION_PR"],range(3200)):
    translation = dic_venue_translations[value]
    cultural_spaces_clean.at[i,"FUNCION_PR"] = translation


types_cultural_spaces=cultural_spaces_clean["FUNCION_PR"].value_counts().to_frame()
types_cultural_spaces.reset_index(inplace=True)
types_cultural_spaces.rename(columns={'index':'TYPE',"FUNCION_PR":"number_cultural_spaces"},inplace=True)
types_cultural_spaces
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="darkgrid")
# %%
f, ax = plt.subplots(figsize=(7, 7))
sns.barplot(x="number_cultural_spaces",y="TYPE",data=types_cultural_spaces,color="g")
# legend
# ax.legend(ncol=2, loc="lower right", frameon=True)
ax.set(xlim=(0, 650), ylabel="",
       xlabel="Number of cultural spaces")
sns.despine(left=True, bottom=True)
plt.savefig('svm_conf.png',bbox_inches='tight', dpi=400)
# %%


barrios_cultural_spaces=cultural_spaces_clean.groupby('BARRIO')['FUNCION_PR'].count().to_frame()
barrios_cultural_spaces.reset_index(inplace=True)
barrios_cultural_spaces.rename(columns={'BARRIO':'neighborhood',"FUNCION_PR":"number_cultural_spaces"},inplace=True)
barrios_cultural_spaces
top_5=barrios_cultural_spaces.sort_values(by=["number_cultural_spaces"],ascending=False).head()
top_5
# %%
f, ax = plt.subplots(figsize=(7, 7))
sns.barplot(y="number_cultural_spaces",x="neighborhood",data=top_5)
ax.set(ylabel="Cultural spaces", xlabel="Barrio or Neighborhood",title="Top 5 Barrios")
sns.despine(left=True, bottom=True)
plt.savefig('svm_conf.png',bbox_inches='tight', dpi=400)
# %%

# Distribucion de espacios culturales por barrios, son 48 barrios.
# num_barrios= 48
# dic_distribucion_per_barrio={"Library":[],
#     "Bookshop":[],
#     "Theater Space":[],
# 	"Exhibition Space":[],
# 	"Cultural Center":[],
# 	"Monuments and Historical Sites":[],
# 	"Bar":[],
# 	"Comprehensive Training Space":[],
# 	"Record Store":[],
# 	"Carousel":[],
# 	"Carnival":[],
# 	"Cinema":[]
# }
# for space in dic_distribucion_per_barrio.keys():
#     list_space = venues_each_neighborhood.xs(space, level="FUNCION_PR").tolist()
#     while(len(list_space)!= num_barrios):
#         list_space.append(0)
#     dic_distribucion_per_barrio[space]=list_space
#
# dic_distribucion_per_barrio["Bookshop"]
venues_each_neighborhood = cultural_spaces_clean.groupby('BARRIO')['FUNCION_PR'].value_counts()
venues_each_neighborhood.to_frame().columns #se tiene que reemplzar el nombre de una columna para evitar error con el indice del mismo nombre
distro_data=venues_each_neighborhood.to_frame().rename(columns={"FUNCION_PR":"cantidad"})
distro_data.reset_index(inplace=True)
distro_data

# %%
f, ax = plt.subplots(figsize=(10, 12))
ax = sns.boxplot(y="FUNCION_PR",x="cantidad",data=distro_data,orient="h")
ax.set(ylabel="", xlabel="",title="Distribution of Cultural Spaces in CABA")
plt.savefig('svm_conf.png',bbox_inches='tight', dpi=400)
# %%
distro_data.sort_values(by=["cantidad"],ascending=False).head(10)
#medianas

distro_data.groupby(['FUNCION_PR'])['cantidad'].median().to_frame().sort_values(by=["cantidad"],ascending=False)



# one hot encoding
caba_onehot = pd.get_dummies(cultural_spaces_clean[cultural_spaces_clean[["FUNCION_PR"]] != "DummySTR"]["FUNCION_PR"], prefix="", prefix_sep="")
# add neighborhood column back to dataframe
caba_onehot.insert(loc=0, column="Neighborhood", value=cultural_spaces_clean[['BARRIO']].values)
caba_onehot.shape
caba_onehot.head()

caba_grouped = caba_onehot.groupby('Neighborhood').mean().reset_index()
caba_grouped_num_cc=caba_onehot.groupby('Neighborhood').sum().reset_index()
caba_grouped_num_cc
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

num_top_venues = 3

indicators = ['st', 'nd', 'rd']
# create columns according to number of top venues
columns = ['Neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common CE'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common CE'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighborhood'] = caba_grouped['Neighborhood']

for ind in np.arange(caba_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(caba_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head(10)
describe_neighborhoods=neighborhoods_venues_sorted.describe(include="all")
describe_neighborhoods
describe_neighborhoods.drop("Neighborhood",axis=1)
barrios_cultural_spaces
barrios_cultural_spaces_sorted=neighborhoods_venues_sorted.join(barrios_cultural_spaces.set_index("neighborhood"),on="Neighborhood")
barrios_cultural_spaces_sorted.rename(columns={"number_cultural_spaces":"Total cultural spaces"},inplace= True)
barrios_cultural_spaces_sorted.sort_values(by=["Total cultural spaces"],ascending=False).head(10)
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
neighborhoods_venues_clusters=neighborhoods_venues_clusters.join(caba_grouped_num_cc.set_index("Neighborhood"), on="Neighborhood")
neighborhoods_venues_clusters



cluster_0=neighborhoods_venues_clusters[neighborhoods_venues_clusters["Cluster Labels"]== 0]
cluster_0.insert(len(cluster_0.columns),'Sum of CE',cluster_0.sum(axis=1))
cluster_0
cluster_0.sort_values(by=["Sum of CE"],ascending=False).head(3)

distro_data
a=distro_data[((distro_data.BARRIO=="PALERMO") | (distro_data.BARRIO=="RECOLETA")|
    (distro_data.BARRIO=="MONSERRAT")) & ((distro_data.FUNCION_PR =="Library")|
    (distro_data.FUNCION_PR =="Bar") | (distro_data.FUNCION_PR =="Bookshop")|
    (distro_data.FUNCION_PR =="Cultural Center") | (distro_data.FUNCION_PR =="Exhibition Space") |
    (distro_data.FUNCION_PR =="Monuments and Historical Sites") | (distro_data.FUNCION_PR =="Theater Space"))
    ]
#%%
f, ax = plt.subplots(figsize=(7, 7))
sns.barplot(y="FUNCION_PR",x="cantidad",hue="BARRIO",data=a)
ax.set(xlabel="Cultural spaces", ylabel="",title="")
sns.despine(left=True, bottom=True)
plt.savefig('svm_conf.png',bbox_inches='tight', dpi=400)
#%%
cultural_spaces.FUNCION_PR.value_counts()
# Cuantos museos tiene Palermo
palermo_museos_galerias=cultural_spaces[(cultural_spaces.BARRIO=="PALERMO") &
    (cultural_spaces.FUNCION_PR=="ESPACIO DE EXHIBICION") & (cultural_spaces.SUBCATEGOR =="MUSEO")
    ]
palermo_museos_galerias.shape

recoleta=cultural_spaces[(cultural_spaces.BARRIO =="RECOLETA") &
    (cultural_spaces.FUNCION_PR=="MONUMENTOS Y LUGARES HISTORICOS")
    ]
recoleta.shape



cluster_0["Sum of CE"].sum()
cluster_0["Library"].sum()
cluster_0["Exhibition Space"].sum()
cluster_0["Monuments and Historical Sites"].sum()
cluster_0.drop(["Cluster Labels","Neighborhood"],axis=1).describe(include="all")

# -------------------------------------------------------
cluster_1=neighborhoods_venues_clusters[neighborhoods_venues_clusters["Cluster Labels"]== 1]
cluster_1.insert(len(cluster_1.columns),'Sum of CE',cluster_1.sum(axis=1))
cluster_1
cluster_1.sort_values(by=["Sum of CE"],ascending=False).head(3)
cluster_1["Sum of CE"].sum()
cluster_1["Library"].sum()
cluster_1["Exhibition Space"].sum()
cluster_1.drop(["Cluster Labels","Neighborhood"],axis=1).describe(include="all")

a=distro_data[((distro_data.BARRIO=="SAN NICOLAS") | (distro_data.BARRIO=="BALVANERA")|
    (distro_data.BARRIO=="BELGRANO")) & ((distro_data.FUNCION_PR =="Library")|(distro_data.FUNCION_PR =="Record Store")|
    (distro_data.FUNCION_PR =="Comprehensive Training Space") | (distro_data.FUNCION_PR =="Bookshop")|
    (distro_data.FUNCION_PR =="Cultural Center") | (distro_data.FUNCION_PR =="Exhibition Space") |
    (distro_data.FUNCION_PR =="Monuments and Historical Sites") | (distro_data.FUNCION_PR =="Theater Space"))
    ]
#%%
f, ax = plt.subplots(figsize=(7, 7))
sns.barplot(y="FUNCION_PR",x="cantidad",hue="BARRIO",data=a)
ax.set(xlabel="Cultural spaces", ylabel="",title="")
sns.despine(left=True, bottom=True)
plt.savefig('svm_conf.png',bbox_inches='tight', dpi=400)
#%%
# -------------------------------------------------------
cluster_2=neighborhoods_venues_clusters[neighborhoods_venues_clusters["Cluster Labels"]== 2]
cluster_2.insert(len(cluster_2.columns),'Sum of CE',cluster_2.sum(axis=1))
cluster_2
cluster_2.sort_values(by=["Sum of CE"],ascending=False).head(3)
cluster_2["Sum of CE"].sum()
cluster_2["Library"].sum()
cluster_2["Exhibition Space"].sum()
cluster_2.drop(["Cluster Labels","Neighborhood"],axis=1).describe(include="all")
subcategorias=cultural_spaces.groupby("BARRIO")["SUBCATEGOR"].value_counts().to_frame()
subcategorias.rename(columns={"SUBCATEGOR":"cantidad"},inplace=True)
subcategorias.reset_index(inplace=True)
subcategorias.sort_values(by=["cantidad"],ascending=False).head(30)
categorias=cultural_spaces_clean.groupby("BARRIO")["FUNCION_PR"].value_counts().to_frame()
categorias.rename(columns={"FUNCION_PR":"cantidad"},inplace=True)
categorias.reset_index(inplace=True)
categorias.sort_values(by=["cantidad"],ascending=False, inplace=True)
categorias.rename(columns={"FUNCION_PR":"Cultural Space","cantidad":"Total"}).head(10)
cultural_spaces.groupby('FUNCION_PR')['SUBCATEGOR'].value_counts()

cultural_spaces[(cultural_spaces["BARRIO"]=="ALMAGRO") & (cultural_spaces["FUNCION_PR"]=="ESPACIO ESCENICO")]

a=distro_data[((distro_data.BARRIO=="ALMAGRO") | (distro_data.BARRIO=="VILLA CRESPO")| 
    (distro_data.BARRIO=="BARRACAS")) & ((distro_data.FUNCION_PR =="Library")|(distro_data.FUNCION_PR =="Bar")|
    (distro_data.FUNCION_PR =="Comprehensive Training Space") | (distro_data.FUNCION_PR =="Bookshop")|
    (distro_data.FUNCION_PR =="Cultural Center") | (distro_data.FUNCION_PR =="Exhibition Space") |
    (distro_data.FUNCION_PR =="Theater Space"))
    ]
#%%
f, ax = plt.subplots(figsize=(7, 7))
sns.barplot(y="FUNCION_PR",x="cantidad",hue="BARRIO",data=a)
ax.set(xlabel="Cultural spaces", ylabel="",title="")
sns.despine(left=True, bottom=True)
plt.savefig('svm_conf.png',bbox_inches='tight', dpi=400)
#%%

















address = 'Buenos Aires'
geolocator = Nominatim(user_agent="CABA")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
barrios_geo = r'barrios.json' # geojson file


caba_map = folium.Map(location=[latitude, longitude],tiles="cartodbpositron", zoom_start=13)
# generate choropleth map using the total immigration of each country to Canada from 1980 to 2013
caba_map.choropleth(
    geo_data=barrios_geo,
    data=neighborhoods_venues_clusters,
    columns=['Neighborhood', 'Cluster Labels'],
    key_on='feature.properties.barrio',
    fill_color='RdYlBu',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='Immigration to Canada',
)

# # display map
caba_map.choropleth()
