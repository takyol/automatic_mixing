from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from automix.data.collate import collate_variable_tracks
from automix.losses.mrstft import MultiResolutionSTFTLoss


def run_epoch(model, loader, loss_fn, optimizer=None, device="cpu"):
    """Runs one pass over `loader`. If `optimizer` is given, trains;
    otherwise runs in eval/no-grad mode. Returns the mean loss."""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    num_batches = 0
    for stems, mask, target in loader:
        stems = stems.to(device)
        mask = mask.to(device)
        target = target.to(device)

        if is_train:
            optimizer.zero_grad()
            pred = model(stems, mask)
            loss = loss_fn(pred, target)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                pred = model(stems, mask)
                loss = loss_fn(pred, target)

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def train(model, train_dataset, val_dataset, num_epochs: int, batch_size: int,
          lr: float, checkpoint_dir: Path, log_dir: Path, device: str = "cpu",
          checkpoint_every: int = 1, num_workers: int = 0):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    print(f"Training on device: {device}")
    model.to(device)
    optimizer = torch.optim.Adam(model.mlp.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=20)
    loss_fn = MultiResolutionSTFTLoss().to(device)

    # Workers must not be persistent: resample() redraws the clip set in the
    # main process each epoch, and only freshly spawned workers pick that up.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                               collate_fn=collate_variable_tracks,
                               num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_variable_tracks,
                             num_workers=num_workers)

    best_val_loss = float("inf")
    for epoch in range(num_epochs):
        train_dataset.resample()
        train_loss = run_epoch(model, train_loader, loss_fn, optimizer=optimizer, device=device)
        val_loss = run_epoch(model, val_loader, loss_fn, optimizer=None, device=device)
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("lr", current_lr, epoch)
        print(f"epoch {epoch + 1}/{num_epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  lr={current_lr:.2e}")

        checkpoint = {
            "epoch": epoch,
            "mlp_state_dict": model.mlp.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": val_loss,
        }
        is_checkpoint_epoch = (epoch + 1) % checkpoint_every == 0 or epoch == num_epochs - 1
        if is_checkpoint_epoch:
            torch.save(checkpoint, checkpoint_dir / "last.pt")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, checkpoint_dir / "best.pt")

    writer.close()
