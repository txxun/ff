# train.py
# main training script
import os
import re
import json
import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from nuscenes.nuscenes import NuScenes

from data import nuScenesDataset, CollateFn


def make_data_loaders(args):
    dataset_kwargs = {
        "n_input": args.n_input,
        "n_samples": args.n_samples,
        "n_output": args.n_output,
        "train_on_all_sweeps": args.train_on_all_sweeps,
    }
    data_loader_kwargs = {
        "pin_memory": True,  # NOTE
        "shuffle": True,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers
    }
    nusc = NuScenes(args.nusc_version, args.nusc_root)
    data_loaders = {
        # consider using all the frames only for training
        "train": DataLoader(nuScenesDataset(nusc, "train", dataset_kwargs),
                            collate_fn=CollateFn, **data_loader_kwargs),
        "val": DataLoader(nuScenesDataset(nusc, "val", dataset_kwargs),
                          collate_fn=CollateFn, **data_loader_kwargs)
    }
    return data_loaders

def mkdir_if_not_exists(d):
    if not os.path.exists(d):
        print(f"creating directory {d}")
        os.makedirs(d)

def resume_from_ckpts(ckpt_dir, model, optimizer, scheduler):
    if len(os.listdir(ckpt_dir)) > 0:
        pattern = re.compile(r"model_epoch_(\d+).pth")
        epochs = []
        for f in os.listdir(ckpt_dir):
            m = pattern.findall(f)
            if len(m) > 0:
                epochs.append(int(m[0]))
        resume_epoch = max(epochs)
        ckpt_path = f"{ckpt_dir}/model_epoch_{resume_epoch}.pth"
        print(f"Resume training from checkpoint {ckpt_path}")

        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        start_epoch = 1 + checkpoint['epoch']
        n_iter = checkpoint["n_iter"]
    else:
        start_epoch = 0
        n_iter = 0
    return start_epoch, n_iter


def data_to_device(batch, device):
    datum = {}
    for key, val in batch.items():
        if isinstance(val, torch.Tensor):
            datum[key] = val.to(device)
        else:
            datum[key] = val
    return datum

def train(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device_count = torch.cuda.device_count()

    #
    data_loaders = make_data_loaders(args)

    # instantiate a model and a renderer
    _n_input, _n_output = args.n_input, args.n_output
    _pc_range, _voxel_size = args.pc_range, args.voxel_size
    if args.model_type == "vanilla":
        from model import VanillaNeuralMotionPlanner
        model = VanillaNeuralMotionPlanner(_n_input, _n_output, _pc_range, _voxel_size)
    elif args.model_type == "vf_guided":
        from model import VFGuidedNeuralMotionPlanner
        model = VFGuidedNeuralMotionPlanner(_n_input, _n_output, _pc_range, _voxel_size)
    elif args.model_type == "obj_guided":
        from model import ObjGuidedNeuralMotionPlanner
        model = ObjGuidedNeuralMotionPlanner(_n_input, _n_output, _pc_range, _voxel_size)
    elif args.model_type == "obj_shadow_guided":
        from model import ObjShadowGuidedNeuralMotionPlanner
        model = ObjShadowGuidedNeuralMotionPlanner(_n_input, _n_output, _pc_range, _voxel_size)
    else:
        raise NotImplementedError(f"{args.model_type} not implemented yet.")

    #
    # model = model.to(device)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_start)

    # scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_epoch, gamma=args.lr_decay)

    # dump config
    mkdir_if_not_exists(args.model_dir)
    with open(f"{args.model_dir}/config.json", 'w') as f:
        json.dump(args.__dict__, f, indent=4)

    # resume
    ckpt_dir = f"{args.model_dir}/ckpts"
    mkdir_if_not_exists(ckpt_dir)
    start_epoch, n_iter = resume_from_ckpts(ckpt_dir, model, optimizer, scheduler)
    device = "cuda:0"
    # data parallel
    # model = nn.DataParallel(model)
    # model = nn.DataParallel(model, device_ids = [4, 5, 6, 7])
    model = model.to(device)

    #
    # writer = SummaryWriter(f"{args.model_dir}/tf_logs")
    for epoch in range(start_epoch, args.num_epoch):
        for phase in ["train", "val"]:
            data_loader = data_loaders[phase]
            if phase == "train":
                print("before train")
                model.train()
                print("after train")
            else:
                model.eval()

            sum_val_loss = {}
            num_batch = len(data_loader)
            print("num_batch ", num_batch)
            num_example = len(data_loader.dataset)
            print("num_example ", num_example)
            for i, batch in enumerate(data_loader):
                print("i ", i)
                bs = len(batch["sample_data_tokens"])
                #print("bs ", bs, ", device_count ", device_count)
                #if bs < device_count:
                #    print(f"Dropping the last batch of size {bs}")
                #    continue
                batch = data_to_device(batch, device)
                # use the following to prevent overfitting
                if args.max_iters_per_epoch > 0 and i >= args.max_iters_per_epoch:
                    print(f"Breaking because of exceeding {args.max_iters_per_epoch} iterations.")
                    break
                print("phase ", phase)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    results = model(batch, "train")
                    loss = results["loss"].mean()
                    print("loss: ", loss)
                    if phase == "train":
                        print("before backward")
                        loss.backward()
                        print("after backward")
                        optimizer.step()
                        print("after step")

                print("Phase: {phase}, Iter: {n_iter},",
                      "Epoch: {epoch}/{args.num_epoch},",
                      "Batch: {i}/{num_batch}",
                      "Loss: {loss.item():.6f}")

                if phase == "train":
                    n_iter += 1
                    print("n_iter: ", n_iter)
                    for key in results:
                        print("before writer")
                        # writer.add_scalar(f"{phase}/{key}", results[key].mean().item(), n_iter)
                        print("after writer")
                else:
                    for key in results:
                        if key not in sum_val_loss:
                            sum_val_loss[key] = 0.0
                        sum_val_loss[key] += (results[key].mean().item() * bs)

            if phase == "train":
                ckpt_path = f"{ckpt_dir}/model_epoch_{epoch}.pth"
                print("ckpt_path: ", ckpt_path)
                torch.save({
                    "epoch": epoch,
                    "n_iter": n_iter,
                    "model_state_dict": model.module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                }, ckpt_path, _use_new_zipfile_serialization=False)
                print("after torch save")
            else:
                for key in sum_val_loss:
                    mean_val_loss = sum_val_loss[key] / num_example
                    # writer.add_scalar(f"{phase}/{key}", mean_val_loss, n_iter)

        print("before scheduler step")
        scheduler.step()
        print("after scheduler step")
    #
    # writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    data_group = parser.add_argument_group("data")
    data_group.add_argument("--dataset", type=str, default="nuscenes")
    data_group.add_argument("--nusc-root", type=str, default="/data/nuscenes")
    data_group.add_argument("--nusc-version", type=str, default="v1.0-trainval")
    data_group.add_argument("--pc-range", type=float, nargs="+", default=[-40.0, -70.4, -2.0, 40.0, 70.4, 3.4])
    data_group.add_argument("--voxel-size", type=float, default=0.2)
    data_group.add_argument("--n-input", type=int, default=20)
    data_group.add_argument("--n-samples", type=int, default=1000)
    data_group.add_argument("--n-output", type=int, default=7)

    model_group = parser.add_argument_group("model")
    model_group.add_argument("--model-type", type=str, required=True)
    model_group.add_argument("--train-on-all-sweeps", action="store_true")
    model_group.add_argument("--flow-mode", type=int, default=3)
    model_group.add_argument("--nvf-loss-factor", type=float, default=1.0)
    model_group.add_argument("--obj-loss-factor", type=float, default=1.0)
    model_group.add_argument("--occ-loss-factor", type=float, default=1.0)
    model_group.add_argument("--model-dir", type=str, required=True)
    model_group.add_argument("--optimizer", type=str, default="Adam")  # Adam with 5e-4
    model_group.add_argument("--lr-start", type=float, default=5e-4)
    model_group.add_argument("--lr-epoch", type=int, default=5)
    model_group.add_argument("--lr-decay", type=float, default=0.1)
    model_group.add_argument("--num-epoch", type=int, default=15)
    model_group.add_argument("--max-iters-per-epoch", type=int, default=-1)
    model_group.add_argument("--batch-size", type=int, default=36)
    model_group.add_argument("--num-workers", type=int, default=18)

    args = parser.parse_args()

    train(args)
