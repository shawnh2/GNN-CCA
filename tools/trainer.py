import time
import json
import os.path as osp

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
from tools.utils import udf_collate_fn, make_dir


class Trainer:

    def __init__(self, args):
        self.args = args
        self.feature_extractor = FeatureExtractor(
            model_name=self.args.reid_name,
            model_path=self.args.reid_path,
            device=self.args.device
        )
        self.node_feature_encoder = NodeFeatureEncoder(self.args.device)
        self.edge_feature_encoder = EdgeFeatureEncoder(self.args.device)
        self.mpn = MPN(self.args.device)
        self.predictor = EdgePredictor(self.args.device)
        self.metrics_name = ['ARI', 'AMI', 'ACC', 'H', 'C', 'V-m']
        make_dir(self.args.output)

    def load_dataset(self):
        dataset_mode = ['train', 'eval'] if self.args.train else ['test']
        if self.args.epfl:
            dataset_name = 'EPFL'
            dataset = [EPFLDataset(self.args.seq_name, mode, self.feature_extractor, f'./dataset/{dataset_name}')
                       for mode in dataset_mode]
        else:
            raise ValueError("Please assign a valid dataset for training!")

        logger.info(f"Loading {dataset_name} sequences {dataset[0].seq_names}")
        logger.info(f"Total graphs for training: {len(dataset[0])} and validating: {len(dataset[1])}"
                    if self.args.train else f"Total graphs for testing: {len(dataset[0])}")
        return dataset

    def train(self):
        output_dir = osp.join(self.args.output, f'train-{int(time.time())}')
        make_dir(output_dir)

        optim = torch.optim.Adam(
            [{"params": self.node_feature_encoder.parameters()},
             {"params": self.edge_feature_encoder.parameters()},
             {"params": self.mpn.parameters()},
             {"params": self.predictor.parameters()}],
            lr=0.01
        )
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.args.epochs, eta_min=0.001)
        scheduler_warmup = GradualWarmupScheduler(optim, 1.0, 10, scheduler_cosine)
        # this zero gradient update is needed to avoid a warning message
        optim.zero_grad()
        optim.step()

        train_set, eval_set = self.load_dataset()
        train_loader = DataLoader(train_set, self.args.batch_size, True, collate_fn=udf_collate_fn, drop_last=True)
        eval_loader = DataLoader(eval_set, self.args.eval_batch_size, collate_fn=udf_collate_fn)
        writer = SummaryWriter(output_dir)
        logger.info("Training begin...")
        for epoch in range(self.args.epochs):
            self._train_one_epoch(epoch, train_loader, optim, scheduler_warmup, writer)
            self._eval_one_epoch(epoch, eval_loader, writer)
            self._save_one_epoch(epoch, output_dir)
        writer.close()

    def test(self):
        ckpt = torch.load(self.args.ckpt)
        self.node_feature_encoder.load_state_dict(ckpt['node_feature_encoder'])
        self.edge_feature_encoder.load_state_dict(ckpt['edge_feature_encoder'])
        self.mpn.load_state_dict(ckpt['mpn'])
        self.predictor.load_state_dict(ckpt['predictor'])

        output_dir = osp.join(self.args.output, f'test-{int(time.time())}')
        visualize_dir = None
        make_dir(output_dir)
        if self.args.visualize:
            visualize_dir = osp.join(output_dir, 'visualize')
            make_dir(visualize_dir)

        test_set = self.load_dataset()[0]
        test_loader = DataLoader(test_set, collate_fn=udf_collate_fn)
        scores = self._test_one_epoch(test_loader, ckpt['L'], visualize_dir)
        result = {metric: float(score) for metric, score in zip(self.metrics_name, scores)}
        result['test-seq'] = test_set.seq_names
        with open(osp.join(output_dir, 'result.json'), 'w') as fp:
            json.dump(result, fp)
        logger.info(f"Test result has been saved in {output_dir} successfully")

    def _train_one_epoch(self, epoch: int, dataloader, optimizer, scheduler, writer):
        scheduler.step()
        losses = []
        for i, data in enumerate(dataloader):
            graph_losses = []
            for graph, node_feature, edge_feature, y_true, _ in data:
                x_node = self.node_feature_encoder(node_feature)
                x_edge = self.edge_feature_encoder(edge_feature)
                step_losses = []
                # Calc loss from each message passing step.
                for _ in range(self.args.max_passing_steps):
                    x_node, x_edge = self.mpn(graph, x_node, x_edge)
                    y_pred = self.predictor(x_edge)
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
            logger.info(f"epoch=({epoch}/{self.args.epochs - 1})"
                        f" | [{i + 1}/{len(dataloader)}]"
                        f" | loss={loss_val:.4f}"
                        f" | avg_graph_loss={loss_val / self.args.batch_size:.4f}")
        avg_loss = sum(losses) / len(losses)
        writer.add_scalar("Loss/train", avg_loss, epoch)
        logger.info(f"finished epoch {epoch}. avg_train_loss={avg_loss:.4f}")

    def _eval_one_epoch(self, epoch: int, dataloader, writer):
        avg_scores = self._test_one_epoch(dataloader, self.args.max_passing_steps)
        log_string = ""
        for i, metric_name in enumerate(self.metrics_name):
            writer.add_scalar(f"Val/{metric_name}", avg_scores[i], epoch)
            log_string += f"{metric_name}={avg_scores[i]:.4f} | "
        logger.info(f"validation results at epoch={epoch}: {log_string}")

    @torch.no_grad()
    def _test_one_epoch(self, dataloader, max_passing_steps: int, visualize_output_dir=None):
        scores_collector = torch.zeros(len(self.metrics_name), dtype=torch.float32)
        n = 0
        for i, data in enumerate(tqdm.tqdm(dataloader)):
            for graph, node_feature, edge_feature, y_true, other in data:
                x_node = self.node_feature_encoder(node_feature)
                x_edge = self.edge_feature_encoder(edge_feature)
                for _ in range(max_passing_steps):
                    x_node, x_edge = self.mpn(graph, x_node, x_edge)
                # Classification on the final message step.
                y_pred = self.predictor(x_edge)
                cluster = ClusterDetections(y_pred, y_true, graph)
                cluster.pruning_and_splitting()
                if visualize_output_dir is not None:
                    sid, fid = other['seq_id'], other['frame_id']
                    output_dir = osp.join(visualize_output_dir, dataloader.dataset.seq_names[sid])
                    make_dir(output_dir)
                    cluster.visualize(
                        dataloader.dataset.load_images(sid, fid, tensor=False),
                        osp.join(output_dir, f'frame_{fid}.jpg')
                    )
                scores = cluster.scores()
                scores_collector += torch.tensor(scores, dtype=torch.float32)
                n += 1
        return scores_collector / n

    def _save_one_epoch(self, epoch: int, output_dir):
        logger.info("Saving model...")
        model_path = osp.join(output_dir,
                              f"gnn_cca_{self.args.device}_epoch_{epoch}.pth.tar")
        torch.save({
            "node_feature_encoder": self.node_feature_encoder.state_dict(),
            "edge_feature_encoder": self.edge_feature_encoder.state_dict(),
            "mpn": self.mpn.state_dict(),
            "predictor": self.predictor.state_dict(),
            "L": self.args.max_passing_steps
        }, model_path)
        logger.info(f"Model has been saved in {model_path}.\n")
