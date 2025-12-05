"""
Utility functions for saving plots to the report figures directory.
"""
import os
import sys

def get_figures_dir():
    """Get the path to the report figures directory."""
    try:
        # Get project root
        if '__file__' in globals():
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
        else:
            project_root = os.path.abspath("../../")
        
        figures_dir = os.path.join(project_root, "refs/report/figures")
        os.makedirs(figures_dir, exist_ok=True)
        return figures_dir
    except:
        # Fallback
        return os.path.join(os.getcwd(), "figures")

def save_plot(fig, filename, format='pdf', dpi=300):
    """
    Save a plotly figure to the figures directory.
    
    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        The figure to save
    filename : str
        Base filename (without extension)
    format : str, default='pdf'
        Format to save ('pdf', 'png', 'html', or 'both')
    dpi : int, default=300
        DPI for PNG format
    """
    figures_dir = get_figures_dir()
    
    # Always save HTML as backup
    html_path = os.path.join(figures_dir, f"{filename}.html")
    fig.write_html(html_path)
    
    if format == 'pdf' or format == 'both':
        pdf_path = os.path.join(figures_dir, f"{filename}.pdf")
        try:
            # Try to save as PDF (requires kaleido)
            fig.write_image(pdf_path, width=1200, height=800, scale=2, format='pdf')
            print(f"  Saved PDF: {pdf_path}")
        except Exception as e:
            print(f"  Note: PDF export not available ({e}), HTML saved instead: {html_path}")
    
    if format == 'png' or format == 'both':
        png_path = os.path.join(figures_dir, f"{filename}.png")
        try:
            fig.write_image(png_path, width=1200, height=800, scale=2, format='png')
            print(f"  Saved PNG: {png_path}")
        except Exception as e:
            print(f"  Note: PNG export not available: {e}")

