import torch


def train_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device
):
    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        _, predicted = torch.max(
            outputs.data,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

    accuracy = 100 * correct / total

    return (
        total_loss / len(loader),
        accuracy
    )


def validate(
    model,
    loader,
    criterion,
    device
):
    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            total_loss += loss.item()

            _, predicted = torch.max(
                outputs.data,
                1
            )

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

    accuracy = 100 * correct / total

    return (
        total_loss / len(loader),
        accuracy
    )


def evaluate_with_predictions(model, loader, device):
    """Runs inference over a loader and returns (predictions, labels) for
    the final classification_report/confusion_matrix in train.py."""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    return all_preds, all_labels


def train_stage(model, train_loader, val_loader, criterion, optimizer,
                device, epochs, best_acc,
                checkpoint_path, class_to_idx, stage_name):
    """Runs `epochs` epochs of train_epoch/validate, saving the checkpoint
    whenever val accuracy improves. Called from train.py for both the
    head-only stage and the fine-tuning stage."""
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=2, factor=0.5
    )

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader,
                                            criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step(val_acc)

        print(
            f"[{stage_name}] epoch {epoch}/{epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.2f}%"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "class_to_idx": class_to_idx,
                "num_classes": len(class_to_idx),
            }, checkpoint_path)
            print(f"  -> new best ({checkpoint_path}), val_acc={val_acc:.2f}%")

    return best_acc
