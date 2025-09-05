# scripts/generate_report.py

import pandas as pd
import os

def generate_html_report(transformed_df, insights_log, predicted_kits, prediction_date, output_path='data/full_report.html'):
    """
    Generates a comprehensive HTML report with all insights and predictions.
    """
    try:
        # Start HTML document
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>RTGS Agent: Comprehensive Health Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 2em; }
                h1, h2 { color: #333; }
                table { width: 100%; border-collapse: collapse; margin-bottom: 2em; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .insights { white-space: pre-wrap; font-family: monospace; }
                .report-section { margin-bottom: 2em; }
            </style>
        </head>
        <body>
            <h1>RTGS Agent: Comprehensive Health Report</h1>
            <p>This report provides a full analysis of the Mother and Child Health Kit scheme data, including key insights, predictions, and a geospatial visualization.</p>
        """
        
        # --- Add Initial Insights (ASCII tables converted to pre-formatted text) ---
        html_content += '<div class="report-section"><h2>Initial Insights</h2><pre class="insights">'
        html_content += insights_log
        html_content += '</pre></div>'
        
        # --- Add Predictive Analysis ---
        html_content += '<div class="report-section"><h2>Predictive Analysis</h2>'
        html_content += f'<p><strong>Predicted MCH kits for {prediction_date}:</strong> {predicted_kits}</p></div>'

        # --- Add Geospatial Map ---
        html_content += '<div class="report-section"><h2>Geospatial Visualization</h2>'
        html_content += '<p>The map below shows the MCH Kit Coverage Ratio by district.</p>'
        # Check if the map file exists before linking it
        if os.path.exists('data/telangana_map.html'):
            html_content += '<iframe src="telangana_map.html" width="100%" height="500"></iframe></div>'
        else:
            html_content += '<p>The map file was not found. Please ensure it was created successfully.</p></div>'

        # End HTML document
        html_content += """
        </body>
        </html>
        """
        
        # Write the HTML content to the file with UTF-8 encoding
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"   - Comprehensive HTML report saved to {output_path}")

    except Exception as e:
        print(f"Error generating HTML report: {e}")