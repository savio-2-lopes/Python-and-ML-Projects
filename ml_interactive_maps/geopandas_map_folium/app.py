import pandas as pd
import geopandas
import folium

urtl = "https://en.wikipedia.org/wiki/List_of_countries_by_meat_consumption"
tables = pd.read_html(url)
table = tables[0]

pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 200)

world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))

table = world.merge(table, how="left", left_on=['name'], right_on=['Country'])
table = table.dropna(subset=['kg/person (2002)[9][note 1]'])
print(table)

my_map = folium.Map()
folium.Choropleth(
    geo_data=table,
    name="choropleth",
    data=table,
    columns=["Country", "kg/person (2002)[9][note 1]"],
    key_on="feature.properties.name",
    fill_color="OrRd",
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name="Meat consumption in kg/person",
).add_to(my_map)

my_map.save('index.html')