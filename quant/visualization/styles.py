"""
Institutional Color Palette
Professional colors for all visualizations following JP Morgan standards.
"""

class InstitutionalColors:
    """Institutional color palette for professional reports."""
    
    # Primary Colors
    PRIMARY_BLUE = '#1a5490'      # JP Morgan blue (main brand color)
    DARK_GRAY = '#4a4a4a'         # Professional dark gray
    GOLD = '#d4af37'              # Institutional gold accent
    
    # Secondary Colors
    SUCCESS_GREEN = '#2e7d32'     # Institutional green
    WARNING_ORANGE = '#f57c00'    # Professional orange
    DANGER_RED = '#c62828'        # Institutional red
    
    # Neutral Colors
    LIGHT_GRAY = '#e0e0e0'        # Light gray for backgrounds
    MEDIUM_GRAY = '#757575'       # Medium gray for secondary text
    WHITE = '#ffffff'
    BLACK = '#000000'
    
    # Chart Colors (for multi-series)
    CHART_COLORS = [
        '#1a5490',  # Primary blue
        '#d4af37',  # Gold
        '#2e7d32',  # Green
        '#f57c00',  # Orange
        '#c62828',  # Red
        '#4a4a4a',  # Dark gray
        '#7b1fa2',  # Purple
        '#0277bd',  # Light blue
    ]
    
    # Heatmap Colors
    HEATMAP_POSITIVE = '#2e7d32'  # Green for positive
    HEATMAP_NEGATIVE = '#c62828'  # Red for negative
    HEATMAP_NEUTRAL = '#757575'   # Gray for neutral
    
    @classmethod
    def get_color_palette(cls, n_colors: int = None):
        """Get color palette for charts."""
        if n_colors is None:
            return cls.CHART_COLORS
        return cls.CHART_COLORS[:n_colors] if n_colors <= len(cls.CHART_COLORS) else cls.CHART_COLORS
    
    @classmethod
    def get_gradient(cls, start_color: str, end_color: str, n_steps: int = 10):
        """Generate gradient between two colors."""
        import matplotlib.colors as mcolors
        import numpy as np
        
        cmap = mcolors.LinearSegmentedColormap.from_list("", [start_color, end_color])
        return [mcolors.rgb2hex(cmap(i)) for i in np.linspace(0, 1, n_steps)]
    
    @classmethod
    def apply_institutional_style(cls, ax):
        """Apply institutional styling to matplotlib axis."""
        # Grid
        ax.grid(True, alpha=0.3, color=cls.MEDIUM_GRAY, linestyle='--', linewidth=0.5)
        
        # Spines
        for spine in ax.spines.values():
            spine.set_color(cls.DARK_GRAY)
            spine.set_linewidth(0.8)
        
        # Tick colors
        ax.tick_params(colors=cls.DARK_GRAY, which='both')
        
        # Labels
        ax.xaxis.label.set_color(cls.DARK_GRAY)
        ax.yaxis.label.set_color(cls.DARK_GRAY)
        ax.title.set_color(cls.PRIMARY_BLUE)
        
        return ax
