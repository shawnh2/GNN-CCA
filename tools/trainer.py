import os
import time

import tqdm
import torch
from loguru import logger
from torchvision.ops import sigmoid_focal_loss
from torchreid.utils import FeatureExtractor
from warmup_scheduler import GradualWarmupScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models import NodeFeatureEncoder, EdgeFeatureEncoder, MPN, EdgePredictor
from tools.dataset import EPFLDataset
from tools.cluster import ClusterDetections
from tools.utils import udf_collate_fn


def train(args):
    if not os.path.exists(args.output):
        os.mkdir(args.output)
    output_dir = os.path.join(args.output, str(int(time.time())))
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    feature_extractor = FeatureExtractor(
        model_name=args.reid_name,
        model_path=args.reid_path,
        device=args.device
    )
    node_feature_encoder = NodeFeatureEncoder(args.device)
    edge_feature_encoder = EdgeFeatureEncoder(args.device)
    mpn = MPN(args.device)
    predictor = EdgePredictor(args.device)
    optim = torch.optim.Adam(
        [{"params": node_feature_encoder.parameters()},
         {"params": edge_feature_encoder.parameters()},
         {"params": mpn.parameters()},
         {"params": predictor.parameters()}],
        lr=0.01
    )
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optim, args.epochs, eta_min=0.001)
    scheduler_warmup = GradualWarmupScheduler(optim, 1.0, 10, scheduler_cosine)
    # this zero gradient update is needed to avoid a warning message
    optim.zero_grad()
    optim.step()

    if args.epfl:
        dataset_dir = './dataset/EPFL'
        train_set = EPFLDataset(args.seq_name, 'train', feature_extractor, dataset_dir)
        eval_set = EPFLDataset(args.seq_name, 'eval', feature_extractor, dataset_dir)
        logger.info(f"Loading EPFL sequences {train_set.seq_names} from {dataset_dir}")
        logger.info(f"Total graphs for training: {len(train_set)} and validating: {len(eval_set)}")
    else:
        raise ValueError("Please assign a valid dataset for training!")

    train_loader = DataLoader(train_set, args.batch_size, True, collate_fn=udf_collate_fn, drop_last=True)
    eval_loader = DataLoader(eval_set, args.eval_batch_size, collate_fn=udf_collate_fn)
    writer = SummaryWriter(output_dir)

    logger.info("Training begin...")
    for epoch in range(args.epochs):
        train_one_epoch(
            epoch, args, train_loader,
            node_feature_encoder, edge_feature_encoder, mpn, predictor,
            optim, scheduler_warmup,
            writer
        )
        eval_one_epoch(
            epoch, args, eval_loader,
            node_feature_encoder, edge_feature_encoder, mpn, predictor,
            writer
        )
        save_one_epoch(
            epoch, args,
            node_feature_encoder, edge_feature_encoder, mpn, predictor,
            output_dir
        )
    writer.close()


def train_one_epoch(epoch: int, args, dataloader,
                    node_feature_encoder, edge_feature_encoder,
                    mpn, edge_predictor,
                    optimizer, scheduler, writer):
    scheduler.step()
    losses = []
    for i, data in enumerate(dataloader):
        graph_losses = []
        for graph, node_feature, edge_feature, y_true in data:
            x_node = node_feature_encoder(node_feature)
            x_edge = edge_feature_encoder(edge_feature)
            step_losses = []
            # Calc loss from each message passing step.
            for _ in range(args.max_passing_steps):
                x_node, x_edge = mpn(graph, x_node, x_edge)
                y_pred = edge_predictor(x_edge)
                # The graph suffers from a serious class-imbalance problem.
                step_loss = sigmoid_focal_loss(y_pred, y_true, 0.9, 5, "mean")
                step_losses.append(step_loss)
            graph_loss = sum(step_losses)
            graph_losses.append(graph_loss)
        loss = sum(graph_losses)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_val = loss.item()
        losses.append(loss_val)
        logger.info(f"epoch=({epoch}/{args.epochs - 1})"
                    f" | batch=({i}/{len(dataloader) - 1})"
                    f" | total_loss={loss_val:.4f}"
                    f" | avg_graph_loss={loss_val / args.batch_size:.4f}")
    avg_loss = sum(losses) / len(losses)
    writer.add_scalar("Loss/train", avg_loss, epoch)
    logger.info(f"finished epoch {epoch}. avg_train_loss={avg_loss:.4f}")


@torch.no_grad()
def eval_one_epoch(epoch: int, args, dataloader,
                   node_feature_encoder, edge_feature_encoder,
                   mpn, edge_predictor,
                   writer):
    metrics_name = ['ARI', 'AMI', 'ACC', 'H', 'C', 'V-m']
    scores_collector = torch.zeros(len(dataloader), 6, dtype=torch.float32)
    for i, data in enumerate(tqdm.tqdm(dataloader)):
        for graph, node_feature, edge_feature, y_true in data:
            x_node = node_feature_encoder(node_feature)
            x_edge = edge_feature_encoder(edge_feature)
            for _ in range(args.max_passing_steps):
                x_node, x_edge = mpn(graph, x_node, x_edge)
            # Classification on the final message step.
            y_pred = edge_predictor(x_edge)
            cluster = ClusterDetections(y_pred, y_true, graph)
            cluster.pruning_and_splitting()
            scores = cluster.scores()
            scores_collector[i] = torch.tensor(scores, dtype=torch.float32)
    avg_scores = torch.mean(scores_collector, 0)
    log_string = ""
    for i, metric_name in enumerate(metrics_name):
        writer.add_scalar(f"Val/{metric_name}", avg_scores[i], epoch)
        log_string += f"{metric_name}={avg_scores[i]:.4f} | "
    logger.info(f"validation results at epoch={epoch}: {log_string}")


def save_one_epoch(epoch: int, args,
                   node_feature_encoder, edge_feature_encoder,
                   mpn, edge_predictor,
                   output_dir):
    logger.info("Saving model...")
    model_path = os.path.join(output_dir,
                              f"gnn_cca_{args.device}_epoch_{epoch}.pth.tar")
    torch.save({
        "node_feature_encoder": node_feature_encoder.state_dict(),
        "edge_feature_encoder": edge_feature_encoder.state_dict(),
        "mpn": mpn.state_dict(),
        "predictor": edge_predictor.state_dict()
    }, model_path)
    logger.info(f"Model has been saved in {model_path}.\n")
