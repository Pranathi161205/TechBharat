# scripts/visualize_data.py

import pandas as pd
import folium

TELANGANA_GEOJSON_PATH = 'data/telangana.geojson'
def create_choropleth_map(df, output_path='data/telangana_map.html'):
    """
    Creates an interactive choropleth map and saves it as an HTML file.
    """
    if df is None or df.empty:
        print("Error: DataFrame is empty or not provided for map creation.")
        return

    try:
        # Create a base map
        m = folium.Map(location=[17.385, 78.4867], zoom_start=8)

        # Create the choropleth map
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
        
        # Save the map
        m.save(output_path)
        print(f"   - Interactive map saved to {output_path}")

    except Exception as e:
        print(f"Error creating map: {e}")