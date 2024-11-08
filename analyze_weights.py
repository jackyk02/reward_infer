import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os

def load_and_analyze_lora(bin_path):
    # Load the model with weights_only=True for safety
    lora_state_dict = torch.load(bin_path, weights_only=True)
    
    # Organize weights by layer type
    layer_weights = defaultdict(list)
    stats = {}
    
    # Analyze each parameter
    for name, param in lora_state_dict.items():
        # Convert bfloat16 to float32 before converting to numpy
        if param.dtype == torch.bfloat16:
            weights = param.to(torch.float32).cpu().numpy()
        else:
            weights = param.cpu().numpy()
            
        # Print parameter info for debugging
        print(f"\nParameter: {name}")
        print(f"Original dtype: {param.dtype}")
        print(f"Shape: {param.shape}")
        
        # Categorize by lora up/down
        if 'lora_A' in name:
            layer_weights['lora_down'].append(weights)
        elif 'lora_B' in name:
            layer_weights['lora_up'].append(weights)
        elif 'alpha' in name or 'rank' in name:
            layer_weights['metadata'].append((name, weights))
            
        # Calculate statistics
        stats[name] = {
            'shape': weights.shape,
            'mean': float(np.mean(weights)),
            'std': float(np.std(weights)),
            'min': float(np.min(weights)),
            'max': float(np.max(weights)),
            'original_dtype': str(param.dtype)
        }
    
    return layer_weights, stats

def visualize_weights(layer_weights, stats, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subplots for different visualizations
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Weight Distribution Histogram
    plt.subplot(2, 2, 1)
    all_weights = []
    for weights in layer_weights['lora_up'] + layer_weights['lora_down']:
        all_weights.extend(weights.flatten())
    plt.hist(all_weights, bins=50, alpha=0.7, color='blue')
    plt.title('Weight Distribution')
    plt.xlabel('Weight Value')
    plt.ylabel('Count')
    
    # 2. Heatmap of first up/down matrices
    if layer_weights['lora_up'] and layer_weights['lora_down']:
        plt.subplot(2, 2, 2)
        up_weights = layer_weights['lora_up'][0]
        plt.imshow(up_weights[:10, :10], cmap='coolwarm', aspect='auto')
        plt.colorbar()
        plt.title('First LoRA Up Matrix (10x10)')
        
        plt.subplot(2, 2, 3)
        down_weights = layer_weights['lora_down'][0]
        plt.imshow(down_weights[:10, :10], cmap='coolwarm', aspect='auto')
        plt.colorbar()
        plt.title('First LoRA Down Matrix (10x10)')
    
    # 3. Statistics summary
    plt.subplot(2, 2, 4)
    plt.axis('off')
    summary_text = "Model Statistics:\n\n"
    summary_text += f"Number of LoRA up matrices: {len(layer_weights['lora_up'])}\n"
    summary_text += f"Number of LoRA down matrices: {len(layer_weights['lora_down'])}\n"
    summary_text += "\nWeight Statistics:\n"
    summary_text += f"Mean range: [{min(s['mean'] for s in stats.values()):.4f}, {max(s['mean'] for s in stats.values()):.4f}]\n"
    summary_text += f"Std range: [{min(s['std'] for s in stats.values()):.4f}, {max(s['std'] for s in stats.values()):.4f}]\n"
    summary_text += "\nData Types:\n"
    dtypes = set(s['original_dtype'] for s in stats.values())
    summary_text += f"Original dtypes: {', '.join(dtypes)}"
    plt.text(0.1, 0.9, summary_text, fontsize=10, verticalalignment='top')
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(output_dir, 'lora_analysis.png')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Plot saved to: {output_path}")
    return fig

# Main analysis function
def analyze_lora_model(bin_path, output_dir='lora_analysis'):
    print(f"Loading LoRA model from {bin_path}")
    layer_weights, stats = load_and_analyze_lora(bin_path)
    
    # Print basic statistics
    print("\nModel Structure:")
    for name, stat in stats.items():
        print(f"\n{name}:")
        print(f"  Original dtype: {stat['original_dtype']}")
        print(f"  Shape: {stat['shape']}")
        print(f"  Mean: {stat['mean']:.6f}")
        print(f"  Std: {stat['std']:.6f}")
        print(f"  Range: [{stat['min']:.6f}, {stat['max']:.6f}]")
    
    # Create and save visualizations
    fig = visualize_weights(layer_weights, stats, output_dir)
    return layer_weights, stats, fig

# Usage example:
bin_model_path = 'adapter_model.bin'
output_dir = 'lora_analysis'  # Directory where the plot will be saved
weights, stats, _ = analyze_lora_model(bin_model_path, output_dir)