"""The huggingface Trainer doesn't really support metrics for
    seq2seq models, so we have to subclass it
    https://github.com/huggingface/transformers/issues/4947
"""


from typing import Optional, List, Dict, Union, Any, NamedTuple
import logging
import os
import pickle

from packaging import version

from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    EvalPrediction,
    PredictionOutput,
    TrainOutput,
    # is_wandb_available,
    set_seed,
)
from transformers.training_args import TrainingArguments

from transformers.file_utils import is_apex_available, is_torch_tpu_available
from transformers import Trainer, DataCollator
from dataclasses import dataclass, field

import numpy as np
from scipy.special import softmax

from tqdm.auto import tqdm, trange


import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, Sampler, SequentialSampler

import torch.nn.functional as f


from src.models.CORAL_BART.utils import safe_decode, write_jsonl, load_pickled_tree
from src.data.scrape_library_structures import TokenizedGraph

import pdb
logger = logging.getLogger(__name__)


def pad_seq(seq: List[int], max_batch_len: int, pad_value: int) -> List[int]:
    return seq + (max_batch_len - len(seq)) * [pad_value]

def load_tokenized_graph(graph_path):
    path = os.path.join(graph_path,"tokenized_graph.pickle")
    f = open(path,"rb")
    graph = pickle.load(f)
    f.close
    return graph


#Maybe theå
class CORALBARTTrainer(Trainer):

    def __init__(self, *args, **kwargs):


        try:
            self.save_eval = kwargs.pop("save_eval")
        except KeyError:
            self.save_eval = False

        try:
            self.tokenizer = kwargs.pop("tokenizer")
        except KeyError:
            self.tokenizer = None

        graph_path = kwargs.pop("library_graph",None)

        if graph_path:
            self.tokenized_graph = load_tokenized_graph(graph_path)
        else:
            self.tokenized_graph = None

        self.graph_loss_weight = kwargs.pop("graph_loss_weight",None)
        self.graph_loss_burn_in_epochs = kwargs.pop("graph_loss_burn_in_epochs",0.)
        
        self.span_aware_loss = kwargs.pop("span_aware_loss",False)
        self.unchanged_loss_weight = kwargs.pop("unchanged_loss_weight",None)

        self.pos_class_weight = kwargs.pop("pos_class_weight",False)
        
        self.oracle_span_aware_decoder = kwargs.pop("oracle_span_aware_decoder",False)
        self.oracle_mixin_p = kwargs.pop("oracle_mixin_p",0.0)

        self.forced_acc = kwargs.pop("forced_acc",False)
        
        self.classification_loss_weight = kwargs.pop("classification_loss_weight",1)
        # if hasattr(self,"library_graph"):
        #     self.library_graph = TokenizedGraph(load_pickled_tree(self.library_graph),self.tokenizer)
        # else:
        #     self.library_graph = None

        super().__init__(*args, **kwargs)


    def train(self, model_path: Optional[str] = None):
        """
        Main training entry point.

        Args:
            model_path (:obj:`str`, `optional`):
                Local path to the model if the model to train has been instantiated from a local path. If present,
                training will resume from the optimizer/scheduler states loaded here.
        """
        train_dataloader = self.get_train_dataloader()
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            num_train_epochs = (
                self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
            )
        else:
            t_total = int(len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs)
            num_train_epochs = self.args.num_train_epochs

        optimizer, scheduler = self.get_optimizers(num_training_steps=t_total)

        # Check if saved optimizer or scheduler states exist
        if (
            model_path is not None
            and os.path.isfile(os.path.join(model_path, "optimizer.pt"))
            and os.path.isfile(os.path.join(model_path, "scheduler.pt"))
        ):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(
                torch.load(os.path.join(model_path, "optimizer.pt"), map_location=self.args.device)
            )
            scheduler.load_state_dict(torch.load(os.path.join(model_path, "scheduler.pt")))

        model = self.model
        if self.args.fp16:
            if not is_apex_available():
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=self.args.fp16_opt_level)

        # multi-gpu training (should be after apex fp16 initialization)
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Distributed training (should be after apex fp16 initialization)
        if self.args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=True,
            )

        if self.tb_writer is not None:
            self.tb_writer.add_text("args", self.args.to_json_string())
            self.tb_writer.add_hparams(self.args.to_sanitized_dict(), metric_dict={})

        # Train!
        if is_torch_tpu_available():
            total_train_batch_size = self.args.train_batch_size * xm.xrt_world_size()
        else:
            total_train_batch_size = (
                self.args.train_batch_size
                * self.args.gradient_accumulation_steps
                * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1)
            )
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", self.num_examples(train_dataloader))
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info("  Instantaneous batch size per device = %d", self.args.per_device_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", total_train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        self.global_step = 0
        self.epoch = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        # Check if continuing training from a checkpoint
        if model_path is not None:
            # set global_step to global_step of last saved checkpoint from model path
            try:
                self.global_step = int(model_path.split("-")[-1].split("/")[0])
                epochs_trained = self.global_step // (len(train_dataloader) // self.args.gradient_accumulation_steps)
                steps_trained_in_current_epoch = self.global_step % (
                    len(train_dataloader) // self.args.gradient_accumulation_steps
                )

                logger.info("  Continuing training from checkpoint, will skip to saved global_step")
                logger.info("  Continuing training from epoch %d", epochs_trained)
                logger.info("  Continuing training from global step %d", self.global_step)
                logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
            except ValueError:
                self.global_step = 0
                logger.info("  Starting fine-tuning.")

        tr_loss = 0.0
        logging_loss = 0.0
        model.zero_grad()
        train_iterator = trange(
            epochs_trained, int(num_train_epochs), desc="Epoch", disable=not self.is_local_master()
        )
        for epoch in train_iterator:
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)

            if is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [self.args.device]).per_device_loader(
                    self.args.device
                )
                epoch_iterator = tqdm(parallel_loader, desc="Iteration", disable=not self.is_local_master())
            else:
                epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=not self.is_local_master())

            # Reset the past mems state at the beginning of each epoch if necessary.
            if self.args.past_index >= 0:
                self._past = None

            for step, inputs in enumerate(epoch_iterator):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue


                loss = self._training_step(model, inputs, optimizer)
                if np.isnan(loss) or np.isinf(loss):
                    logger.warning(f"Invalid loss: {loss}. Zeroing grad and continuing.")
                    model.zero_grad()
                    continue


                tr_loss += loss

                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    len(epoch_iterator) <= self.args.gradient_accumulation_steps
                    and (step + 1) == len(epoch_iterator)
                ):
                    if self.args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), self.args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

                    if is_torch_tpu_available():
                        xm.optimizer_step(optimizer)
                    else:
                        optimizer.step()

                    scheduler.step()
                    model.zero_grad()
                    self.global_step += 1
                    self.epoch = epoch + (step + 1) / len(epoch_iterator)

                    if (self.args.logging_steps > 0 and self.global_step % self.args.logging_steps == 0) or (
                        self.global_step == 1 and self.args.logging_first_step
                    ):
                        logs: Dict[str, float] = {}
                        logs["loss"] = (tr_loss - logging_loss) / self.args.logging_steps
                        # backward compatibility for pytorch schedulers
                        logs["learning_rate"] = (
                            scheduler.get_last_lr()[0]
                            if version.parse(torch.__version__) >= version.parse("1.4")
                            else scheduler.get_lr()[0]
                        )
                        logging_loss = tr_loss

                        self._log(logs)

                    if self.args.evaluate_during_training and self.global_step % self.args.eval_steps == 0:
                        self.evaluate()

                    if self.args.save_steps > 0 and self.global_step % self.args.save_steps == 0:
                        # In all cases (even distributed/parallel), self.model is always a reference
                        # to the model we want to save.
                        if hasattr(model, "module"):
                            assert model.module is self.model
                        else:
                            assert model is self.model
                        # Save model checkpoint
                        output_dir = os.path.join(self.args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{self.global_step}")

                        self.save_model(output_dir)

                        if self.is_world_master():
                            self._rotate_checkpoints()

                        if is_torch_tpu_available():
                            xm.rendezvous("saving_optimizer_states")
                            xm.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                            xm.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        elif self.is_world_master():
                            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

                if self.args.max_steps > 0 and self.global_step > self.args.max_steps:
                    epoch_iterator.close()
                    break
            if self.args.max_steps > 0 and self.global_step > self.args.max_steps:
                train_iterator.close()
                break
            if self.args.tpu_metrics_debug or self.args.debug:
                # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                xm.master_print(met.metrics_report())

        if self.tb_writer:
            self.tb_writer.close()
        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        return TrainOutput(self.global_step, tr_loss / self.global_step)

class CORALBARTTrainerSeq2Seq(CORALBARTTrainer):

    def __init__(self, *args, **kwargs):

        try:
            self.generation_kwargs = kwargs.pop("generation_kwargs")
        except KeyError:

            self.generation_kwargs = {"num_beams": 5,
                                      "max_length": 256, "early_stopping": True}

        try:
            self.face = kwargs.pop("face")
        except KeyError:
            self.face = False
        
        super().__init__(*args, **kwargs)

        if self.face:
            self.loss_fct = self.face_loss
            self.output_token_frequencies = np.zeros(len(self.tokenizer))
        else:
            self.loss_fct = self.ce_loss


        self.generation_kwargs["inserted_id"] = self.tokenizer.convert_tokens_to_ids("<INSERTED>")
        self.generation_kwargs["end_of_inserted_id"] = self.tokenizer.convert_tokens_to_ids("</INSERTED>")
        self.generation_kwargs["eos_token_id"] = self.tokenizer.eos_token_id
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)

    def face_loss(self,logits,labels,training = False):
        #Implements the FACE-OPR model from "Improving Neural Diversity with Frequency-Aware Loss"
        if training:
            self.update_output_token_counts(logits)
        self.criterion.weight = self.loss_weight()
        return self.ce_loss(logits,labels,training=False)

    def ce_loss(self,logits,labels, training = False):
        return self.criterion(logits,labels)

    def update_output_token_counts(self,logits):
        greedy_predictions = torch.argmax(logits,axis=1).cpu().tolist()
        for pred in greedy_predictions:
            if not pred in self.tokenizer.all_special_ids:
                self.output_token_frequencies[pred] += 1


    def loss_weight(self):
        RF = self.output_token_frequencies / self.output_token_frequencies.sum() # relative frequency
        a = -1 / RF.max()
        weight = a * RF + 1
        weight = weight / weight.sum() * len(weight) # normalization

        if self.model.device.type == "cuda":
            return torch.FloatTensor(weight).cuda()
        else:
            return torch.FloatTensor(weight)

    # def write_results(self,input_ids,decoder_inputs, preds, classification_labels, loss_masks,
    #                   out_path):
        
    #     results = []
    #     data_iter = zip(input_ids, decoder_inputs, preds, classification_labels, loss_masks, classification_logits)
    #     for input, label, pred, class_labels, loss_mask, class_logits in data_iter:
    #         # pdb.set_trace()
    #         result = {
    #             "input": safe_decode(self.tokenizer, input),
    #             "label": safe_decode(self.tokenizer, label),
    #             "prediction": safe_decode(self.tokenizer, pred),
    #             "input_ids": input,
    #             "label_ids": label,
    #             "pred": pred,
    #             "loss_mask":loss_mask,
    #             "classification_labels":class_labels,
                
    #         }
    #         results.append(result)

    #     with open(out_path, "w") as result_file:
    #         write_jsonl(result_file, results)

    def write_results(self,input_ids,decoder_inputs, preds, out_path, loss_masks):
        results = []
        if len(loss_masks) == 0:
            loss_masks = [[]] * len(input_ids)
        for input, label, pred, loss_mask in zip(input_ids, decoder_inputs, preds, loss_masks):
            # pdb.set_trace()
            result = {
                "input": safe_decode(self.tokenizer, input),
                "label": safe_decode(self.tokenizer, label),
                "prediction": safe_decode(self.tokenizer, pred),
                "input_ids": input,
                "label_ids": label,
                "pred": pred,
                "loss_mask":loss_mask
            }
            results.append(result)

        with open(out_path, "w") as result_file:
            write_jsonl(result_file, results)


    def seq2seq_loss(self,logits, labels, training=False, selector=None, gamma=1.5):
        """Loss is calculated as:
        Args:
            logits ([type]): [description]
            labels ([type]): [description]
            training (bool, optional): [description]. Defaults to False.
            mask ([type], optional): [description]. Defaults to None.
            mask_weight (int, optional): Unmasked labels have a `masked_weight` multiple
                        assigned to their loss. Defaults to 3.
        Returns:
            Tensor: loss
        """
        logits = logits.view(-1, logits.shape[-1])
        labels = labels.view(-1)

        if selector is None or torch.sum(selector) == 0:
            loss = self.loss_fct(logits, labels, training=training)
            return loss
        
        else:
            selector = selector.view(-1)

            selected_indices = torch.where(selector)
            # pdb.set_trace()
            unselected_indices = torch.where(selector==0)

            selected_logits = logits[selected_indices]
            selected_labels = labels[selected_indices]
            

            if self.span_aware_loss:
                loss = self.loss_fct(selected_logits, selected_labels, training=training)
            elif self.unchanged_loss_weight:
                loss = self.loss_fct(selected_logits, selected_labels, training=training) +\
                    (self.unchanged_loss_weight * self.loss_fct(logits, labels, training=training))
            else:
                loss = (gamma-1) * self.loss_fct(selected_logits, selected_labels, training=training) +\
                    self.loss_fct(logits, labels, training=training)

            return loss


    def _prediction_loop(
        self, dataloader: DataLoader, description: str, prediction_loss_only: Optional[bool] = None
    ) -> PredictionOutput:
        """s
        Prediction/evaluation loop, shared by `evaluate()` and `predict()`.
        Works both with or without labels.
        """

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else self.prediction_loss_only

        model = self.model
        # multi-gpu eval
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
        else:
            model = self.model

        # Note: in torch.distributed mode, there's no point in wrapping the model
        # inside a DistributedDataParallel as we'll be under `no_grad` anyways.

        batch_size = dataloader.batch_size
        logger.info("***** Running %s *****", description)
        logger.info("  Num examples = %d", self.num_examples(dataloader))
        logger.info("  Batch size = %d", batch_size)
        eval_losses: List[float] = []
        preds: torch.Tensor = None
        label_ids: torch.Tensor = None
        model.eval()

        input_id_lists = []
        label_id_lists = []
        loss_mask_lists = []
        input_token_label_lists = []
        

        if self.args.past_index >= 0:
            past = None

        for inputs in tqdm(dataloader, desc=description):
            has_labels = any(inputs.get(k) is not None for k in [
                             "labels", "lm_labels", "masked_lm_labels"])

            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(self.args.device)
            if self.args.past_index >= 0:
                inputs["mems"] = past

            with torch.no_grad():

                source_ids, source_mask, target_ids = inputs["input_ids"], inputs[
                    "attention_mask"], inputs["decoder_input_ids"]

                # Why this line?
                decoder_input_ids = target_ids[:, :-1].contiguous()
                labels = target_ids[:, 1:].clone()
                loss_mask = inputs.get("loss_mask")
                if loss_mask is not None:
                    loss_mask = loss_mask[:, 1:].clone()
                    loss_mask_lists += loss_mask.cpu().tolist()
                    input_token_label_lists += []

                outputs = model(source_ids, attention_mask=source_mask,
                                decoder_input_ids=decoder_input_ids, use_cache=False)
                logits = outputs[0]

                loss = self.seq2seq_loss(logits,labels, selector = loss_mask)
                eval_losses.append(loss.item())

                if self.args.past_index >= 0:
                    past = outputs[self.args.past_index if has_labels else self.args.past_index - 1]

            if not prediction_loss_only:
                if preds is None:
                    preds = model.generate(
                        inputs['input_ids'], **self.generation_kwargs).detach().cpu().tolist()
                else:
                    new_preds = model.generate(
                        inputs['input_ids'], **self.generation_kwargs).detach().cpu().tolist()
                    preds += new_preds

                input_id_lists += inputs["input_ids"].detach().cpu().tolist()

                if len(label_id_lists) == 0:
                    label_id_lists = labels.detach().cpu().tolist()
                else:
                    label_id_lists += labels.detach().cpu().tolist()

        if self.save_eval and not prediction_loss_only:
            eval_out_path = os.path.join(
                self.args.output_dir, "eval-preds-{}.jsonl".format(self.global_step))
            # pdb.set_trace()
            self.write_results(
                input_id_lists, label_id_lists, preds, eval_out_path,loss_mask_lists)

        # Finally, turn the aggregated tensors into numpy arrays.
        if self.compute_metrics is not None and preds is not None:
            metrics = self.compute_metrics(EvalPrediction(
                predictions=preds, label_ids=label_id_lists))

        else:
            metrics = {}
        if len(eval_losses) > 0:
            # This won't work, I think
            metrics["eval_loss"] = np.mean(eval_losses)

        # Prefix all keys with eval_
        for key in list(metrics.keys()):
            if not key.startswith("eval_"):
                metrics[f"eval_{key}"] = metrics.pop(key)

        return PredictionOutput(predictions=preds, label_ids=label_id_lists, metrics=metrics)

    def _training_step(
        self, model: torch.nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], optimizer: torch.optim.Optimizer
    ) -> float:
        model.train()
        # pdb.set_trace()
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.args.device)
        # print(input["item"])
        if "adj_mat" in inputs:

            source_ids, source_mask, target_ids, adj_mat = inputs["input_ids"], inputs[
                "attention_mask"], inputs["decoder_input_ids"], inputs["adj_mat"]
        else:
            source_ids, source_mask, target_ids = inputs["input_ids"], inputs[
                "attention_mask"], inputs["decoder_input_ids"]
            adj_mat = None
        # TODO: Don't understand why we're slicing the input and output like this
        decoder_input_ids = target_ids[:, :-1].contiguous()  # Why this line?
        labels = target_ids[:, 1:].clone()

        loss_mask = inputs.get("loss_mask")
        if loss_mask is not None:
            loss_mask = loss_mask[:, 1:].clone()

        if self.args.past_index >= 0 and self._past is not None:
            inputs["mems"] = self._past

        outputs = model(source_ids, attention_mask=source_mask, decoder_input_ids=decoder_input_ids, use_cache=False)

        logits = outputs[0]
        loss = self.seq2seq_loss(logits, labels, training=True, selector=loss_mask)

        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        else:
            loss.backward()

        return loss.item()



class MultiTaskEvalPrediction(NamedTuple):
    """
    Evaluation output (always contains labels), to be used to compute metrics.
    Parameters:
        predictions (:obj:`np.ndarray`): Predictions of the model.
        label_ids (:obj:`np.ndarray`): Targets to be matched.
    """

    predictions: np.ndarray
    label_ids: np.ndarray
    input_labels : list
    input_logits : list
    input_ids : list
    loss_mask : list

def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    # if y is None:
    #     dist = dist - torch.diag(dist.diag)
    return torch.clamp(dist, 0.0, np.inf)

    for i, row in enumerate(x.split(1)):
        r_v = row.expand_as(y)
        sq_dist = torch.sum((r_v - y) ** 2, 1)
        dist_mat[i] = sq_dist.view(1, -1)
    return dist_mat


def sq_dist(x):
    assert len(x.size()) == 2
    norm = (x ** 2).sum(1).view(-1, 1)
    dn = (norm + norm.view(1, -1)) - 2.0 * (x @ x.t())
    return dn

def pdist(sample_1, sample_2, norm=2, eps=1e-5):
    n_1, n_2 = sample_1.size(0), sample_2.size(0)
    norm = float(norm)
    if norm == 2.:
        norms_1 = torch.sum(sample_1**2, dim=1, keepdim=True)
        norms_2 = torch.sum(sample_2**2, dim=1, keepdim=True)
        norms = (norms_1.expand(n_1, n_2) +
             norms_2.transpose(0, 1).expand(n_1, n_2))
        distances_squared = norms - 2 * sample_1.mm(sample_2.t())
        return distances_squared

def my_cdist(x1, x2):
    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
    res = torch.addmm(x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
    res = res.clamp_min_(1e-5)
    return res

class CORALBARTMultiTaskTrainer(CORALBARTTrainerSeq2Seq):
    def __init__(self,*args, **kwargs):
        # self.classification_loss_weight = kwargs.pop("classification_loss_weight")
        super().__init__(*args,**kwargs)
        class_weights = torch.FloatTensor([1.0,self.pos_class_weight]).cuda()
        self.classifcation_criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        # self.classifcation_criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    def classification_loss(self,logits,labels, selector = None):
        logits = logits.view(-1, logits.shape[-1])
        labels = labels.view(-1)

        if selector is not None:
            selector = selector.view(-1)
            selected_indices = torch.where(selector==1)
            logits = logits[selected_indices]
            labels = labels[selected_indices]

        return self.classifcation_criterion(logits,labels)
    
    def write_results(self,input_ids,decoder_inputs, preds, classification_labels, loss_masks,
                    out_path, classification_logits= None):
        
        if classification_logits is None:
            classification_logits = [None] * len(input_ids)
        results = []
        data_iter = zip(input_ids, decoder_inputs, preds, classification_labels, loss_masks, classification_logits)
        for input, label, pred, class_labels, loss_mask, class_logits in data_iter:
            # pdb.set_trace()
            result = {
                "input": safe_decode(self.tokenizer, input),
                "label": safe_decode(self.tokenizer, label),
                "prediction": safe_decode(self.tokenizer, pred),
                "input_ids": input,
                "label_ids": label,
                "pred": pred,
                "loss_mask":loss_mask,
                "classification_labels":class_labels,
                "classification_logits":class_logits
                
            }
            results.append(result)

        with open(out_path, "w") as result_file:
            write_jsonl(result_file, results)

    def graph_loss(self,input_ids,input_embeddings,scale_factor=0.1, selector=None):
        """Implements the loss from Gu et. al , ICLR 2019 using a
        pre-provided graph of library structure

        Picked scale factor becase on initialization the mean euclidean distance
        was ~10x the mean graph distance


        I think the problem here is because of zeros in the distance?
        Should only compare inputs that have a distance

        Maybe get G_d, get all rows where
        """
        flattened_embeddings = input_embeddings.contiguous().view(-1, input_embeddings.shape[-1])
        flattened_input_ids = input_ids.view(-1)

        if not selector is None:
            selector = selector.view(-1)
            flattened_embeddings = flattened_embeddings[torch.where(selector==1)]
            flattened_input_ids = flattened_input_ids[torch.where(selector==1)]

        supported_mask = np.where([x.item() in self.tokenized_graph.supported_input_ids for x in flattened_input_ids])

        loss_embeddings = flattened_embeddings[supported_mask]
        loss_input_ids = flattened_input_ids[supported_mask]


        g_D = self.tokenized_graph.get_pairwise_graph_distances(loss_input_ids.detach().tolist())
        g_D = loss_embeddings.new(g_D).view(-1)


        # e_D = torch.cdist(loss_embeddings,loss_embeddings,p=1)
        e_D = my_cdist(loss_embeddings,loss_embeddings)
        # Scale distance because embeddings are in such a high dimension and
        # expected distance is high
        e_D = (scale_factor * e_D).view(-1)

        # Gradient is unstable where graph distance is zero
        distance_ratio = e_D[g_D!=0]/g_D[g_D!=0]
        n = distance_ratio.shape[0]

        loss = torch.sum(torch.abs(distance_ratio -1 ))  / (n)
        return loss

    def _training_step(
        self, model: torch.nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], optimizer: torch.optim.Optimizer
    ) -> float:
        model.train()

        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.args.device)

        source_ids, source_mask, target_ids = inputs["input_ids"], inputs["attention_mask"], inputs["decoder_input_ids"]
        input_labels = inputs["input_labels"]

        #TODO: Don't understand why we're slicing the input and output like this
        decoder_input_ids = target_ids[:, :-1].contiguous()  # Why this line?
        labels = target_ids[:, 1:].clone()

        loss_mask = inputs.get("loss_mask")
        if loss_mask is not None:
            loss_mask = loss_mask[:, 1:].clone()

        if self.args.past_index >= 0 and self._past is not None:
            inputs["mems"] = self._past

        #This is necessary because model also runs a dense layer on top of the encoder hidden states
        outputs, input_class_logits, encoder_final_hidden_states = model(source_ids, attention_mask=source_mask, decoder_input_ids=decoder_input_ids, use_cache=False,
                                                    return_input_logits=True, return_final_encoder_hidden_states=True)

        logits = outputs[0]
        seq2seq_loss = self.seq2seq_loss(logits, labels, training=True, selector=loss_mask)
        if self.span_aware_loss:
            classification_loss = self.classification_loss(input_class_logits,input_labels, selector=input_labels)
        else:
            classification_loss = self.classification_loss(input_class_logits,input_labels)


        loss = seq2seq_loss + (self.classification_loss_weight * classification_loss)

        if self.tokenized_graph and self.epoch > self.graph_loss_burn_in_epochs:
            if self.span_aware_loss:
                graph_loss = self.graph_loss(source_ids, encoder_final_hidden_states, selector=input_labels)
            else:
                graph_loss = self.graph_loss(source_ids, encoder_final_hidden_states)
            loss = loss + (self.graph_loss_weight * graph_loss)

        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        else:
            loss.backward()

        return loss.item()


    def _prediction_loop(
            self, dataloader: DataLoader, description: str, prediction_loss_only: Optional[bool] = None
        ) -> PredictionOutput:
        """s
        Prediction/evaluation loop, shared by `evaluate()` and `predict()`.
        Works both with or without labels.
        """
        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else self.prediction_loss_only

        model = self.model
        # multi-gpu eval
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
        else:
            model = self.model

        # Note: in torch.distributed mode, there's no point in wrapping the model
        # inside a DistributedDataParallel as we'll be under `no_grad` anyways.

        batch_size = dataloader.batch_size
        logger.info("***** Running %s *****", description)
        logger.info("  Num examples = %d", self.num_examples(dataloader))
        logger.info("  Batch size = %d", batch_size)
        eval_losses: List[float] = []
        preds: torch.Tensor = None
        label_ids: torch.Tensor = None
        model.eval()

        input_id_lists = []
        label_id_lists = []
        input_token_label_lists = []
        input_token_logit_lists  = []
        loss_mask_lists = []

        if self.args.past_index >= 0:
            past = None

        for inputs in tqdm(dataloader, desc=description):
            has_labels = any(inputs.get(k) is not None for k in ["labels", "lm_labels", "masked_lm_labels"])

            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(self.args.device)
            if self.args.past_index >= 0:
                inputs["mems"] = past

            with torch.no_grad():
                source_ids, source_mask, target_ids = inputs["input_ids"], inputs["attention_mask"], inputs["decoder_input_ids"]
                input_labels = inputs["input_labels"]

                decoder_input_ids = target_ids[:, :-1].contiguous()  # Why this line?
                labels = target_ids[:, 1:].clone()
                loss_mask = inputs.get("loss_mask")
                if loss_mask is not None:
                    loss_mask = loss_mask[:, 1:].clone()

                outputs, input_class_logits, encoder_final_hidden_states = model(source_ids, attention_mask=source_mask, decoder_input_ids=decoder_input_ids, use_cache=False,
                                                    return_input_logits=True, return_final_encoder_hidden_states=True)

                logits = outputs[0]
                if self.span_aware_loss:
                    seq2seq_loss = self.seq2seq_loss(logits, labels, training=False, selector=loss_mask)
                    classification_loss = self.classification_loss(input_class_logits,input_labels,selector=input_labels)
                else:
                    seq2seq_loss = self.seq2seq_loss(logits, labels, training=False)
                    classification_loss = self.classification_loss(input_class_logits,input_labels)


                loss = seq2seq_loss + (self.classification_loss_weight * classification_loss)

                if self.epoch and self.tokenized_graph and self.epoch > self.graph_loss_burn_in_epochs:
                    graph_loss = self.graph_loss(source_ids, encoder_final_hidden_states, selector=input_labels)
                    loss = loss + (self.graph_loss_weight * graph_loss)

                eval_losses.append(loss.item())
                
                input_token_label_lists += input_labels.tolist()
                input_token_logit_lists += input_class_logits.tolist()
                loss_mask_lists += loss_mask.tolist()

                if self.args.past_index >= 0:
                    past = outputs[self.args.past_index if has_labels else self.args.past_index - 1]

            if not prediction_loss_only:
                if self.oracle_span_aware_decoder:
                    oracle_token_labels = input_labels
                elif self.forced_acc:
                    ground_truth_inputs = input_labels.cpu().numpy()
                    random_mask = np.random.random(ground_truth_inputs.shape)>self.forced_acc
                    mask = np.logical_and(random_mask,ground_truth_inputs!=-100)
                    ground_truth_inputs[mask] = 1 - ground_truth_inputs[mask]
                    oracle_token_labels = input_labels.new(ground_truth_inputs)
                else:                    
                    oracle_token_labels = None

                if preds is None:
                    preds = model.generate(inputs['input_ids'], original_inputs=inputs['input_ids'],
                                        oracle_token_labels=oracle_token_labels, oracle_mixin_p = self.oracle_mixin_p,
                                        **self.generation_kwargs).detach().cpu().tolist()
                else:
                    new_preds = model.generate(inputs['input_ids'], original_inputs=inputs['input_ids'], 
                                        oracle_token_labels=oracle_token_labels, oracle_mixin_p = self.oracle_mixin_p, 
                                        **self.generation_kwargs).detach().cpu().tolist()
                    preds += new_preds

                input_id_lists += inputs["input_ids"].detach().cpu().tolist()


                if len(label_id_lists) == 0 :
                    label_id_lists = labels.detach().cpu().tolist()
                else:
                    label_id_lists += labels.detach().cpu().tolist()

        if self.save_eval and not prediction_loss_only:
            eval_out_path = os.path.join(self.args.output_dir, "eval-preds-{}.jsonl".format(self.global_step))
            logger.info(f"Logging results to {eval_out_path}")
            self.write_results(input_ids = input_id_lists,
                               decoder_inputs = label_id_lists, 
                               preds = preds,  
                               classification_labels= input_token_label_lists,
                               out_path = eval_out_path,
                               loss_masks=loss_mask_lists,
                               classification_logits=input_token_logit_lists)


        # Finally, turn the aggregated tensors into numpy arrays.
        if self.compute_metrics is not None and preds is not None:
            metrics = self.compute_metrics(MultiTaskEvalPrediction(predictions=preds,
                                                                   label_ids=label_id_lists,
                                                                   input_ids = input_id_lists,
                                                                   input_labels = input_token_label_lists,
                                                                   input_logits = input_token_logit_lists,
                                                                   loss_mask = loss_mask_lists))
        else:
            metrics = {}
        if len(eval_losses) > 0:
            #This won't work, I think
            metrics["eval_loss"] = np.mean(eval_losses)

        # Prefix all keys with eval_
        for key in list(metrics.keys()):
            if not key.startswith("eval_"):
                metrics[f"eval_{key}"] = metrics.pop(key)

        #TODO: Could also be nice to output token-level classifiction results
        return PredictionOutput(predictions=preds, label_ids=label_id_lists, metrics=metrics)




    def demo_predict(self, test_dataset):
        description="Demo Prediction"
        dataloader = self.get_test_dataloader(test_dataset)
        model = self.model
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
        else:
            model = self.model

        batch_size = dataloader.batch_size
        logger.info("***** Running %s *****", description)
        logger.info("  Num examples = %d", self.num_examples(dataloader))
        logger.info("  Batch size = %d", batch_size)
        eval_losses: List[float] = []
        preds = None
        input_ids = None
        label_ids = None
        model.eval()


        if is_torch_tpu_available():
            dataloader = pl.ParallelLoader(
                dataloader, [self.args.device]).per_device_loader(self.args.device)


        if self.args.past_index >= 0:
            past = None

        results = {"input_ids":[],
                    "generation":[],
                    "classification":[],
                    "class_logits":[]}

        for inputs in tqdm(dataloader, desc=description):
            has_labels = any(inputs.get(k) is not None for k in ["labels", "lm_labels", "masked_lm_labels"])
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(self.args.device)
            if self.args.past_index >= 0:
                inputs["mems"] = past

            with torch.no_grad():
                source_ids, source_mask, target_ids = inputs["input_ids"], inputs["attention_mask"], inputs["decoder_input_ids"]
                input_labels = inputs["input_labels"]

                decoder_input_ids = target_ids[:, :-1].contiguous()  # Why this line?
                labels = target_ids[:, 1:].clone()
                loss_mask = inputs.get("loss_mask")
                if loss_mask is not None:
                    loss_mask = loss_mask[:, 1:].clone()

                outputs, input_class_logits = model(source_ids, attention_mask=source_mask, decoder_input_ids=decoder_input_ids, use_cache=False,
                                                    return_input_logits=True)

                logits = outputs[0]

            results["input_ids"]+=inputs['input_ids'].tolist()
            results["generation"]+=logits.argmax(dim=2).tolist()
            results["classification"]+=input_class_logits.argmax(dim=2).tolist()
            results["class_logits"]+=input_class_logits.tolist()

        return results



class ClassificationTaskEvalPrediction(NamedTuple):
    """
    Evaluation output (always contains labels), to be used to compute metrics.
    Parameters:
        predictions (:obj:`np.ndarray`): Predictions of the model.
        label_ids (:obj:`np.ndarray`): Targets to be matched.
    """

    predictions: list
    input_labels : list
    input_ids : list

class CORALBARTTrainerClassification(CORALBARTTrainer):
    def write_results(self,input_ids, labels, preds, loss_mask, out_path):
        results = []
        for input, label, pred, loss_mask in zip(input_ids, labels, preds, loss_mask):

            result = {
                "input": safe_decode(self.tokenizer, input),
                "classification_labels": label,
                "classification_logits": pred,
                "loss_mask" : loss_mask
            }
            results.append(result)

        with open(out_path, "w") as result_file:
            write_jsonl(result_file, results)
    
    # def train(self, model_path: Optional[str] = None):
    #     return Trainer.train(self)

    def _training_step(self, model, inputs, optimizer):
        inputs = {
            "input_ids": inputs["input_ids"],
            "labels": inputs["input_labels"],
            "attention_mask": inputs["attention_mask"]
        }
        return super()._training_step(model,inputs, optimizer)
    def _prediction_loop(
        self, dataloader: DataLoader, description: str, prediction_loss_only: Optional[bool] = None
    ) -> PredictionOutput:
        """
        Prediction/evaluation loop, shared by `evaluate()` and `predict()`.
        Works both with or without labels.
        """
        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else self.prediction_loss_only

        model = self.model
        # multi-gpu eval
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
        else:
            model = self.model
        # Note: in torch.distributed mode, there's no point in wrapping the model
        # inside a DistributedDataParallel as we'll be under `no_grad` anyways.

        batch_size = dataloader.batch_size
        logger.info("***** Running %s *****", description)
        logger.info("  Num examples = %d", self.num_examples(dataloader))
        logger.info("  Batch size = %d", batch_size)
        eval_losses: List[float] = []
        preds = None
        input_ids = None
        label_ids = None
        loss_mask_lists = []
        model.eval()

        if is_torch_tpu_available():
            dataloader = pl.ParallelLoader(
                dataloader, [self.args.device]).per_device_loader(self.args.device)

        if self.args.past_index >= 0:
            past = None

        for inputs in tqdm(dataloader, desc=description):
            has_labels = any(inputs.get(k) is not None for k in [
                             "labels", "lm_labels", "masked_lm_labels","input_labels"])

            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(self.args.device)
            if self.args.past_index >= 0:
                inputs["mems"] = past

            with torch.no_grad():
                outputs = model(input_ids = inputs["input_ids"], labels=inputs["input_labels"], attention_mask=inputs["attention_mask"])
                if has_labels:
                    step_eval_loss, logits = outputs[:2]
                    eval_losses += [step_eval_loss.mean().item()]
                else:
                    logits = outputs[0]
                if self.args.past_index >= 0:
                    past = outputs[self.args.past_index if has_labels else self.args.past_index - 1]

            if not prediction_loss_only:

                if preds is None:
                    preds = logits.cpu().tolist()
                else:
                    preds += logits.cpu().tolist()
                if inputs.get("input_labels") is not None:
                    if label_ids is None:
                        label_ids = inputs["input_labels"].cpu().tolist()
                        input_ids = inputs["input_ids"].cpu().tolist()
                        loss_mask_lists = inputs["loss_mask"].cpu().tolist()
                    else:

                        label_ids += inputs["input_labels"].detach().tolist()
                        input_ids += inputs["input_ids"].detach().tolist()
                        loss_mask_lists += inputs["loss_mask"].cpu().tolist()

        if self.args.local_rank != -1:
            raise NotImplementedError
            # In distributed mode, concatenate all results from all nodes:
            if preds is not None:
                preds = self.distributed_concat(
                    preds, num_total_examples=self.num_examples(dataloader))
            if label_ids is not None:
                label_ids = self.distributed_concat(
                    label_ids, num_total_examples=self.num_examples(dataloader))

        # Convert preds and labels into numpy
        # if preds is not None:
        #     preds = np.array(preds)
        # if label_ids is not None:
        #     label_ids = np.array(label_ids)

        if self.save_eval and not prediction_loss_only:
            eval_out_path = os.path.join(
                self.args.output_dir, "eval-preds-{}.jsonl".format(self.global_step))
            self.write_results(input_ids, label_ids, preds, loss_mask_lists,eval_out_path)

        if self.compute_metrics is not None and preds is not None and label_ids is not None:
            metrics = self.compute_metrics(ClassificationTaskEvalPrediction(
                    predictions=preds, input_labels =label_ids, input_ids = input_ids))
        else:
            metrics = {}
        if len(eval_losses) > 0:
            metrics["eval_loss"] = np.mean(eval_losses)

        # Prefix all keys with eval_
        for key in list(metrics.keys()):
            if not key.startswith("eval_"):
                metrics[f"eval_{key}"] = metrics.pop(key)

        return PredictionOutput(predictions=preds, label_ids=label_ids, metrics=metrics)
