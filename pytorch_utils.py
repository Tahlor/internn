from pathlib import Path
import torch
from general_tools.utils import get_root

def save_model(path, model, optimizer, epoch, loss):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)

def load_model(path, model, optimizer):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss

def incrementer(root, base, make_new_folder=False):
    """

    Args:
        root (folder):
        base (filename or subfolder):
        make_new_folder (make new subfolder if it doesn't exist): if you just want a new filename, set to false

    Returns:

    """
    new_folder = Path(root) / base
    increment = 0
    base = Path(base)
    while new_folder.exists():
        increment += 1
        increment_string = f"_{increment:02d}" if increment > 0 else ""
        new_folder = Path(root / (base.stem + increment_string + base.suffix))
    if make_new_folder:
        new_folder.mkdir(parents=True, exist_ok=True)
    return new_folder

def get_latest_file(folder, suffix=".pt"):
    list_of_paths = folder.glob('*'+suffix)
    return max(list_of_paths, key=lambda p: p.stat().st_ctime)
