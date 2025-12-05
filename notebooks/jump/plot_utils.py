"""
Utility functions for saving plots to the report figures directory.
"""
import os
import sys
import matplotlib.pyplot as plt

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
    Save a matplotlib figure to the figures directory.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save
    filename : str
        Base filename (without extension)
    format : str, default='pdf'
        Format to save ('pdf', 'png', or 'both')
    dpi : int, default=300
        DPI for PNG format
    """
    figures_dir = get_figures_dir()
    
    if format == 'pdf' or format == 'both':
        pdf_path = os.path.join(figures_dir, f"{filename}.pdf")
        try:
            fig.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=dpi)
            print(f"  Saved PDF: {pdf_path}")
        except Exception as e:
            print(f"  Warning: Could not save PDF ({e})")
    
    if format == 'png' or format == 'both':
        png_path = os.path.join(figures_dir, f"{filename}.png")
        try:
            fig.savefig(png_path, format='png', bbox_inches='tight', dpi=dpi)
            print(f"  Saved PNG: {png_path}")
        except Exception as e:
            print(f"  Warning: Could not save PNG: {e}")
