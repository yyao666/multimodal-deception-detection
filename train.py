import os
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.parsing import parse_protocol_csv
from data.dataset import ParsedFaceSpecDataset, multimodal_collate_fn
from models.multimodal_r50 import MultimodalR50


def train_one_epoch(data_loader, model, optimizer, loss_fn, device):
    """
    Train the model for one epoch.
    """
    model.train()
    epoch_losses = []
    correct = 0
    total = 0

    for _, spec, face, labels in data_loader:
        spec = spec.to(device)
        face = face.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        preds = model(spec, face)
        loss = loss_fn(preds, labels)
        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())
        correct += (torch.argmax(preds, dim=-1) == labels).sum().item()
        total += len(labels)

    return float(np.mean(epoch_losses)), round(correct / total, 4) * 100


def build_clip_level_targets(parsed_test_txt):
    """
    Build clip-level labels and count the number of segments per clip.
    """
    clip_level_names = []
    with open(parsed_test_txt, "r") as f:
        for line in f.readlines():
            name = line.split(",")[0].split("/")[1]
            if name not in clip_level_names:
                clip_level_names.append(name)

    clip_level_labels = np.zeros(len(clip_level_names), dtype=np.int8)
    splits_per_clip = np.zeros(len(clip_level_names), dtype=np.int8)

    with open(parsed_test_txt, "r") as f:
        for line in f.readlines():
            gt = line.split(",")[1]
            label = 0 if gt == "T" else 1
            idx = clip_level_names.index(line.split(",")[0].split("/")[1])

            clip_level_labels[idx] = label
            splits_per_clip[idx] += 1

    return clip_level_names, clip_level_labels, splits_per_clip


def validate_one_epoch(data_loader, model, loss_fn, device, parsed_test_txt):
    """
    Validate the model for one epoch using clip-level aggregation.
    """
    model.eval()
    epoch_losses = []

    clip_level_names, clip_level_labels, splits_per_clip = build_clip_level_targets(parsed_test_txt)
    clip_level_preds = np.zeros(len(clip_level_names), dtype=np.int8)

    with torch.no_grad():
        for sample_names, spec, face, labels in data_loader:
            spec = spec.to(device)
            face = face.to(device)
            labels = labels.to(device)

            preds = model(spec, face)
            loss = loss_fn(preds, labels)
            epoch_losses.append(loss.item())

            for sample_name, pred in zip(sample_names, preds):
                idx = clip_level_names.index(sample_name)
                clip_level_preds[idx] += torch.argmax(pred).item()

    avg_preds = np.round(clip_level_preds / splits_per_clip).astype(np.int8)
    correct = np.sum(clip_level_labels == avg_preds)
    acc = round(correct / len(clip_level_names), 2) * 100

    return float(np.mean(epoch_losses)), acc


def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    torch.manual_seed(config["seed"])
    device = torch.device(config["device"])

    for fold in range(1, config["num_folds"] + 1):
        print(f"\nFold {fold}")

        train_csv = os.path.join(config["protocol_dir"], f"train{fold}.csv")
        test_csv = os.path.join(config["protocol_dir"], f"test{fold}.csv")

        parsed_train_txt = os.path.join(config["parsed_protocol_dir"], "parsed_train.txt")
        parsed_test_txt = os.path.join(config["parsed_protocol_dir"], "parsed_test.txt")
        parsed_train_csv = os.path.join(config["parsed_protocol_dir"], "parsed_train.csv")
        parsed_test_csv = os.path.join(config["parsed_protocol_dir"], "parsed_test.csv")

        os.makedirs(config["parsed_protocol_dir"], exist_ok=True)

        parse_protocol_csv(
            annotations_file=train_csv,
            output_txt=parsed_train_txt,
            output_csv=parsed_train_csv,
            img_dir=config["img_dir"],
            time_window=config["time_window"],
            mode=config["mode"],
        )

        parse_protocol_csv(
            annotations_file=test_csv,
            output_txt=parsed_test_txt,
            output_csv=parsed_test_csv,
            img_dir=config["img_dir"],
            time_window=config["time_window"],
            mode=config["mode"],
        )

        train_dataset = ParsedFaceSpecDataset(
            annotations_file=parsed_train_csv,
            spec_dir=config["spec_dir"],
            img_dir=config["img_dir"],
            time_window=config["time_window"],
            fps=config["fps"],
        )

        test_dataset = ParsedFaceSpecDataset(
            annotations_file=parsed_test_csv,
            spec_dir=config["spec_dir"],
            img_dir=config["img_dir"],
            time_window=config["time_window"],
            fps=config["fps"],
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=config["num_workers"],
            collate_fn=multimodal_collate_fn,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
            collate_fn=multimodal_collate_fn,
        )

        model = MultimodalR50()

        if config["use_data_parallel"]:
            model = nn.DataParallel(model, device_ids=config["device_ids"])

        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
        loss_fn = nn.CrossEntropyLoss()

        best_val_acc = []

        for epoch in range(config["num_epochs"]):
            print(f"\nEpoch {epoch + 1}")

            train_loss, train_acc = train_one_epoch(
                train_loader, model, optimizer, loss_fn, device
            )

            val_loss, val_acc = validate_one_epoch(
                test_loader, model, loss_fn, device, parsed_test_txt
            )

            print(f"Train loss = {train_loss:.4f}, Train accuracy = {train_acc:.2f}")
            print(f"Val loss = {val_loss:.4f}, Val accuracy = {val_acc:.2f}")

            best_val_acc.append(val_acc)

        print(f"Best Accuracy = {max(best_val_acc):.2f}")


if __name__ == "__main__":
    main()
