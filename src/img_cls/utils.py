import torch
import matplotlib
import matplotlib.pyplot as plt
import os

matplotlib.style.use('ggplot')

class SaveBestModel:
    """
    Save best checkpoint according to a monitored metric.
    - mode='min': lower is better (e.g. loss)
    - mode='max': higher is better (e.g. accuracy / macro F1)
    """
    def __init__(self, mode='min', metric_name='validation loss'):
        if mode not in {'min', 'max'}:
            raise ValueError(f"Unsupported mode={mode}. Use 'min' or 'max'.")
        self.mode = mode
        self.metric_name = metric_name
        if self.mode == 'min':
            self.best_value = float('inf')
        else:
            self.best_value = -float('inf')
        
    def is_better(self, current_value):
        if self.mode == 'min':
            return current_value < self.best_value
        return current_value > self.best_value

    def __call__(self, current_value, epoch, model, out_dir, name):
        if self.is_better(current_value):
            self.best_value = current_value
            print(f"\nBest {self.metric_name}: {self.best_value}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                }, os.path.join(out_dir, 'best_'+name+'.pth'))
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.head.state_dict(),
                }, os.path.join(out_dir, 'best_head_'+name+'.pth'))

def save_model(epochs, model, optimizer, criterion, out_dir, name):
    """
    Function to save the trained model to disk.
    """
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, os.path.join(out_dir, name+'.pth'))
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.head.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, os.path.join(out_dir, 'head_'+name+'.pth'))

def save_plots(
    train_acc,
    valid_acc,
    train_loss,
    valid_loss,
    out_dir,
    train_macro_f1=None,
    valid_macro_f1=None,
):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # Accuracy plots.
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='tab:blue', linestyle='-', 
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='tab:red', linestyle='-', 
        label='validataion accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(out_dir, 'accuracy.png'))
    
    # Loss plots.
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='tab:blue', linestyle='-', 
        label='train loss'
    )
    plt.plot(
        valid_loss, color='tab:red', linestyle='-', 
        label='validataion loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(out_dir, 'loss.png'))

    if train_macro_f1 is not None and valid_macro_f1 is not None:
        plt.figure(figsize=(10, 7))
        plt.plot(
            train_macro_f1, color='tab:blue', linestyle='-',
            label='train macro F1'
        )
        plt.plot(
            valid_macro_f1, color='tab:red', linestyle='-',
            label='validation macro F1'
        )
        plt.xlabel('Epochs')
        plt.ylabel('Macro F1')
        plt.legend()
        plt.savefig(os.path.join(out_dir, 'macro_f1.png'))
