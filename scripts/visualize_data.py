# scripts/visualize_data.py

import pandas as pd
import folium

TELANGANA_GEOJSON_PATH = 'data/telangana.json' 

def create_choropleth_map(df):
    if df is None or df.empty:
        print("Error: DataFrame is empty or not provided for map creation.")
        return
    try:
        m = folium.Map(location=[17.385, 78.4867], zoom_start=8)
        folium.Choropleth(
            geo_data=TELANGANA_GEOJSON_PATH,
            name='choropleth',
            data=df,
            columns=['districtName', 'kit_coverage_ratio'],
            key_on='feature.properties.district',
            fill_color='YlGn',
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name='MCH Kit Coverage Ratio'
        ).add_to(m)
        m.save('data/telangana_map.html')
        print(f"   - Interactive map saved to {'data/telangana_map.html'}")
    except Exception as e:
        print(f"Error creating map: {e}")