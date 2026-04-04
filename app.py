from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io
import base64
import json
from datetime import datetime
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

# Configure matplotlib for better visuals
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

class DrugGraphGenerator:
    @staticmethod
    def create_safety_score_chart(interactions_data):
        """Generate bar chart of safety scores"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        drug_pairs = [f"{i['drugA']} + {i['drugB']}" for i in interactions_data]
        scores = [94 if i['safety'] == 'safe' else 28 for i in interactions_data]
        colors = ['#10b981' if i['safety'] == 'safe' else '#f97316' for i in interactions_data]
        
        bars = ax.bar(drug_pairs, scores, color=colors, edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Safety Score (0-100)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Drug Combinations', fontsize=12, fontweight='bold')
        ax.set_title('Clinical Safety Assessment Dashboard', fontsize=14, fontweight='bold', pad=20)
        ax.set_ylim(0, 100)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                   f'{score}', ha='center', va='bottom', fontweight='bold')
        
        # Add legend
        safe_patch = mpatches.Patch(color='#10b981', label='Safe/Low Risk')
        unsafe_patch = mpatches.Patch(color='#f97316', label='High Risk/Caution')
        ax.legend(handles=[safe_patch, unsafe_patch], loc='upper right')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Convert to base64
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
        img.seek(0)
        plt.close()
        
        return base64.b64encode(img.getvalue()).decode()
    
    @staticmethod
    def create_radar_chart(drug_attributes):
        """Generate radar chart for drug comparison"""
        categories = ['Safety', 'Efficacy', 'Metabolism', 'Half-life', 'Interactions']
        num_vars = len(categories)
        
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        for drug in drug_attributes:
            values = drug['scores']
            values += values[:1]  # Close the loop
            ax.plot(angles, values, 'o-', linewidth=2, label=drug['name'])
            ax.fill(angles, values, alpha=0.25)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=10)
        ax.set_ylim(0, 100)
        ax.set_title('Drug Comparison Radar Chart', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
        img.seek(0)
        plt.close()
        
        return base64.b64encode(img.getvalue()).decode()
    
    @staticmethod
    def create_timeline_chart(history_data):
        """Generate timeline of interactions over time"""
        if not history_data:
            return None
        
        dates = []
        safe_count = []
        unsafe_count = []
        
        for h in history_data[:10]:
            try:
                # Parse date string
                date_str = h.get('date', '')
                if date_str:
                    # Handle different date formats
                    if ',' in date_str:
                        date_part = date_str.split(',')[0]
                        dates.append(datetime.strptime(date_part, '%m/%d/%Y'))
                    else:
                        dates.append(datetime.now())
                else:
                    dates.append(datetime.now())
                
                safe = sum(1 for r in h.get('results', []) if r.get('safety') == 'safe')
                unsafe = sum(1 for r in h.get('results', []) if r.get('safety') == 'unsafe')
                safe_count.append(safe)
                unsafe_count.append(unsafe)
            except:
                continue
        
        if not dates:
            return None
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = range(len(dates))
        width = 0.35
        
        bars1 = ax.bar([i - width/2 for i in x], safe_count, width, label='Safe Interactions', color='#10b981')
        bars2 = ax.bar([i + width/2 for i in x], unsafe_count, width, label='High Risk Interactions', color='#f97316')
        
        ax.set_xlabel('Check Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Interactions', fontsize=12, fontweight='bold')
        ax.set_title('Drug Interaction Timeline Analysis', fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels([d.strftime('%Y-%m-%d') for d in dates], rotation=45, ha='right')
        ax.legend()
        
        plt.tight_layout()
        
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
        img.seek(0)
        plt.close()
        
        return base64.b64encode(img.getvalue()).decode()
    
    @staticmethod
    def create_pie_chart(interactions_data):
        """Generate pie chart for interaction types"""
        safe_count = sum(1 for i in interactions_data if i['safety'] == 'safe')
        unsafe_count = sum(1 for i in interactions_data if i['safety'] == 'unsafe')
        unknown_count = sum(1 for i in interactions_data if i['safety'] == 'unknown')
        
        labels = ['Safe/Low Risk', 'High Risk/Caution', 'Unknown']
        sizes = [safe_count, unsafe_count, unknown_count]
        colors = ['#10b981', '#f97316', '#8b5cf6']
        explode = (0.05, 0.1, 0.05)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, colors=colors,
                                           autopct='%1.1f%%', shadow=True, startangle=90)
        
        ax.set_title('Interaction Risk Distribution', fontsize=14, fontweight='bold', pad=20)
        
        # Enhance text
        for text in texts:
            text.set_fontweight('bold')
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        plt.tight_layout()
        
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
        img.seek(0)
        plt.close()
        
        return base64.b64encode(img.getvalue()).decode()
    
    @staticmethod
    def create_heatmap(drug_matrix):
        """Generate correlation heatmap for drugs"""
        if not drug_matrix or len(drug_matrix) < 2:
            return None
        
        # Extract unique drug names
        drug_names = set()
        for key in drug_matrix.keys():
            drugs = key.split('|')
            drug_names.add(drugs[0])
            drug_names.add(drugs[1])
        drug_names = sorted(list(drug_names))
        
        if len(drug_names) < 2:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create matrix
        matrix_data = []
        for d1 in drug_names:
            row = []
            for d2 in drug_names:
                if d1 == d2:
                    row.append(100)
                else:
                    pair_key = '|'.join(sorted([d1, d2]))
                    row.append(drug_matrix.get(pair_key, 50))
            matrix_data.append(row)
        
        im = ax.imshow(matrix_data, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=100)
        
        # Show all ticks
        ax.set_xticks(np.arange(len(drug_names)))
        ax.set_yticks(np.arange(len(drug_names)))
        ax.set_xticklabels(drug_names, rotation=45, ha='right')
        ax.set_yticklabels(drug_names)
        
        # Add text annotations
        for i in range(len(drug_names)):
            for j in range(len(drug_names)):
                text = ax.text(j, i, matrix_data[i][j],
                             ha="center", va="center", color="black", fontweight='bold')
        
        ax.set_title('Drug Interaction Heatmap\n(High Score = Safer)', fontsize=14, fontweight='bold', pad=20)
        plt.colorbar(im, ax=ax, label='Safety Score')
        
        plt.tight_layout()
        
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
        img.seek(0)
        plt.close()
        
        return base64.b64encode(img.getvalue()).decode()

@app.route('/')
def index():
    """Serve the main HTML file"""
    return render_template('index.html')

@app.route('/api/generate_graphs', methods=['POST'])
def generate_graphs():
    """API endpoint to generate graphs from interaction data"""
    try:
        data = request.json
        interactions = data.get('interactions', [])
        
        graphs = {}
        
        # Generate safety score bar chart
        if interactions:
            graphs['safety_chart'] = DrugGraphGenerator.create_safety_score_chart(interactions)
            graphs['pie_chart'] = DrugGraphGenerator.create_pie_chart(interactions)
        
        # Generate radar chart if drug attributes available
        drug_attrs = data.get('drug_attributes', [])
        if drug_attrs and len(drug_attrs) >= 2:
            graphs['radar_chart'] = DrugGraphGenerator.create_radar_chart(drug_attrs)
        
        # Generate timeline from history
        history = data.get('history', [])
        if history:
            timeline = DrugGraphGenerator.create_timeline_chart(history)
            if timeline:
                graphs['timeline_chart'] = timeline
        
        # Generate heatmap
        drug_matrix = data.get('drug_matrix', {})
        if drug_matrix and len(drug_matrix) >= 1:
            heatmap = DrugGraphGenerator.create_heatmap(drug_matrix)
            if heatmap:
                graphs['heatmap'] = heatmap
        
        return jsonify({'success': True, 'graphs': graphs})
    
    except Exception as e:
        print(f"Error generating graphs: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)

