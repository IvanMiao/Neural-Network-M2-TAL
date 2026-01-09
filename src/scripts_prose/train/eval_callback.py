import os
os.environ["KERAS_BACKEND"] = "torch"
import keras
import torch
import json
from datetime import datetime
import matplotlib.pyplot as plt

class TrainingReportCallback(keras.callbacks.Callback):
    
    def __init__(self, output_dir="./reports", model_name="model"):
        super().__init__()
        self.output_dir = output_dir
        self.model_name = model_name
        self.history = {}
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for key, value in logs.items():
            if key not in self.history:
                self.history[key] = []

            self.history[key].append(float(value))
    
    def on_train_end(self, logs=None):
        os.makedirs(self.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{self.model_name}_{timestamp}"
        
        json_path = os.path.join(self.output_dir, f"{base_name}_history.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)
        print(f"Training history saved to {json_path}")
        
        self._plot_metrics(base_name)
        
        self._generate_report(base_name)
    
    def _plot_metrics(self, base_name):
        epochs = range(1, len(self.history.get('loss', [])) + 1)
        if not epochs:
            return
            
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Training Report - {self.model_name}', fontsize=14, fontweight='bold')
        
        # Loss
        ax = axes[0, 0]
        ax.plot(epochs, self.history['loss'], 'b-', label='Train Loss', linewidth=2)
        if 'val_loss' in self.history:
            ax.plot(epochs, self.history['val_loss'], 'r--', label='Val Loss', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Accuracy
        ax = axes[0, 1]
        if 'accuracy' in self.history:
            ax.plot(epochs, self.history['accuracy'], 'b-', label='Train Acc', linewidth=2)
        if 'val_accuracy' in self.history:
            ax.plot(epochs, self.history['val_accuracy'], 'r--', label='Val Acc', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Perplexity
        ax = axes[1, 0]
        if 'perplexity' in self.history:
            ax.plot(epochs, self.history['perplexity'], 'b-', label='Train PPL', linewidth=2)
        if 'val_perplexity' in self.history:
            ax.plot(epochs, self.history['val_perplexity'], 'r--', label='Val PPL', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Perplexity')
        ax.set_title('Perplexity')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Learning Rate
        ax = axes[1, 1]
        if 'learning_rate' in self.history or 'lr' in self.history:
            lr_key = 'learning_rate' if 'learning_rate' in self.history else 'lr'
            ax.plot(epochs, self.history[lr_key], 'g-', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Learning Rate')
            ax.set_title('Learning Rate')
            ax.grid(True, alpha=0.3)
        else:
            ax.plot(epochs, self.history['loss'], 'b-', label='Train', linewidth=2)
            if 'val_loss' in self.history:
                ax.plot(epochs, self.history['val_loss'], 'r--', label='Val', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Train vs Validation Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        chart_path = os.path.join(self.output_dir, f"{base_name}_charts.png")
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Training charts saved to {chart_path}")
    
    def _generate_report(self, base_name):
        report_path = os.path.join(self.output_dir, f"{base_name}_report.md")
        
        final_loss = self.history.get('loss', [0])[-1]
        final_val_loss = self.history.get('val_loss', [0])[-1]
        final_acc = self.history.get('accuracy', [0])[-1]
        final_val_acc = self.history.get('val_accuracy', [0])[-1]
        final_ppl = self.history.get('perplexity', [0])[-1]
        final_val_ppl = self.history.get('val_perplexity', [0])[-1]
        best_val_loss = min(self.history.get('val_loss', [float('inf')]))
        best_epoch = self.history.get('val_loss', []).index(best_val_loss) + 1 if 'val_loss' in self.history else 0
        
        report = f"""# Training Report: {self.model_name}

**Date**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Total Epochs**: {len(self.history.get('loss', []))}

## Final Metrics

| Metric | Train | Validation |
|--------|-------|------------|
| Loss | {final_loss:.4f} | {final_val_loss:.4f} |
| Accuracy | {final_acc:.4f} | {final_val_acc:.4f} |
| Perplexity | {final_ppl:.2f} | {final_val_ppl:.2f} |

## Best Performance

- **Best Val Loss**: {best_val_loss:.4f} (Epoch {best_epoch})

## Training Charts

![Training Charts](./{base_name}_charts.png)
"""
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Training report saved to {report_path}")
