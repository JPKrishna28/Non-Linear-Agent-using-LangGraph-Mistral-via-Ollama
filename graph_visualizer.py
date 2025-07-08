import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import logging
from logger_config import setup_logging

logger = setup_logging()

def visualize_graph(save_path: str = "graph_structure.png"):
    """Create a visual representation of the graph structure"""
    try:
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))

        # Define node positions
        positions = {
            'START': (0, 0),
            'router': (2, 0),
            'math_node': (4, 2),
            'writer_node': (4, 0),
            'translator_node': (4, -2),
            'default_node': (4, -4),
            'final_node': (6, 0),
            'END': (8, 0)
        }

        # Define node colors
        colors = {
            'START': 'lightgreen',
            'router': 'lightblue',
            'math_node': 'lightcoral',
            'writer_node': 'lightyellow',
            'translator_node': 'lightpink',
            'default_node': 'lightgray',
            'final_node': 'lightcyan',
            'END': 'lightgreen'
        }

        # Draw nodes
        for node, (x, y) in positions.items():
            bbox = FancyBboxPatch(
                (x - 0.8, y - 0.3), 1.6, 0.6,
                boxstyle="round,pad=0.1",
                facecolor=colors[node],
                edgecolor='black',
                linewidth=2
            )
            ax.add_patch(bbox)
            ax.text(x, y, node, ha='center', va='center', fontsize=10, fontweight='bold')

        # Draw edges
        edges = [
            ('START', 'router'),
            ('router', 'math_node'),
            ('router', 'writer_node'),
            ('router', 'translator_node'),
            ('router', 'default_node'),
            ('math_node', 'final_node'),
            ('writer_node', 'final_node'),
            ('translator_node', 'final_node'),
            ('default_node', 'final_node'),
            ('final_node', 'END')
        ]

        for start, end in edges:
            start_pos = positions[start]
            end_pos = positions[end]
            ax.annotate('', xy=end_pos, xytext=start_pos,
                        arrowprops=dict(arrowstyle='->', lw=2, color='blue'))

        # Add conditional edges (dashed)
        conditional_edges = [
            ('math_node', 'writer_node'),
            ('math_node', 'translator_node'),
            ('writer_node', 'translator_node')
        ]

        for start, end in conditional_edges:
            start_pos = positions[start]
            end_pos = positions[end]
            ax.annotate('', xy=end_pos, xytext=start_pos,
                        arrowprops=dict(arrowstyle='->', lw=2, color='red', linestyle='dashed'))

        ax.set_xlim(-1, 9)
        ax.set_ylim(-5, 3)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('LangGraph Router System Architecture', fontsize=16, fontweight='bold')

        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], color='blue', lw=2, label='Direct Flow'),
            plt.Line2D([0], [0], color='red', lw=2, linestyle='dashed', label='Conditional Flow')
        ]
        ax.legend(handles=legend_elements, loc='upper right')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Graph visualization saved to {save_path}")
        return True

    except Exception as e:
        logger.error(f"Error creating graph visualization: {e}")
        return False