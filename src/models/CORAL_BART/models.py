from transformers import (BartForConditionalGeneration, BartConfig)
from transformers.generation_utils import BeamHypotheses
import torch
import torch.nn.functional as F
from torch import nn
from src.models.CORAL_BART.utils import load_pickled_tree, separate_regions
import glob
import numpy as np

import os
import sys
from collections import defaultdict

import logging
import pathlib
logger = logging.getLogger(__name__)

root_dir = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(root_dir, "hyperbolics", "pytorch"))
# import hyperbolic_models


def mixin_with_p(a,b,p):
    assert a.shape == b.shape
    mask = np.where(np.random.random(a.shape) < p)
    new_b = np.copy(b)
    new_b[mask] = a[mask]
    return new_b

class MultiTaskBart(BartForConditionalGeneration):
    def __init__(self, config: BartConfig, span_aware_decoding=False):
        super().__init__(config)
        self.token_classifier = nn.Linear(
            config.hidden_size, config.num_labels)
        self.token_classifier_dropout = nn.Dropout(config.hidden_dropout_prob)

        self.span_aware_decoding = config.span_aware_decoding
        self.classification_threshold =  config.classification_threshold

    def forward(
        self,
        input_ids,
        attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        decoder_cached_states=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_input_logits=False,
        return_final_encoder_hidden_states=False,
        **unused,
    ):
        seq2seq_outputs = super().forward(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            decoder_cached_states=decoder_cached_states,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=return_input_logits
        )
        
        to_return = [seq2seq_outputs]
        
        if not return_input_logits:
            return seq2seq_outputs
        
        encoder_final_hidden_states = seq2seq_outputs[-2]
        if return_input_logits:  
            sequence_output = self.token_classifier_dropout(
                encoder_final_hidden_states)
            token_logits = self.token_classifier(sequence_output)
            to_return.append(token_logits)

        if return_final_encoder_hidden_states:
            to_return.append(encoder_final_hidden_states)
        
        return tuple(to_return)

    def classify_ids_api(self, input_ids):
        # Classifies input ids using the encoder. Also ultiumately runs
        # the decoder, so this could probably be optimized.
        _, token_logits = self(input_ids, return_input_logits=True)
        scores = torch.softmax(token_logits, axis=2)
        classes = (scores[:, :, -1] >= self.classification_threshold).int()
        return classes

    def _generate_beam_search(
        self,
        input_ids,
        cur_len,
        max_length,
        min_length,
        do_sample,
        early_stopping,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        no_repeat_ngram_size,
        bad_words_ids,
        pad_token_id,
        eos_token_id,
        batch_size,
        num_return_sequences,
        length_penalty,
        num_beams,
        vocab_size,
        encoder_outputs,
        attention_mask,
        use_cache,
        model_specific_kwargs,
    ):
        """ Generate sequences for each example with beam search.
        """

        # generated hypotheses
        generated_hyps = [
            BeamHypotheses(num_beams, max_length, length_penalty,
                           early_stopping=early_stopping)
            for _ in range(batch_size)
        ]

        # scores for each sentence in the beam
        beam_scores = torch.zeros(
            (batch_size, num_beams), dtype=torch.float, device=input_ids.device)

        # for greedy decoding it is made sure that only tokens of the first beam are considered to avoid sampling the exact same tokens three times
        if do_sample is False:
            beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)  # shape (batch_size * num_beams,)

        # cache compute states
        past = (encoder_outputs, None) if encoder_outputs is not None else None

        # done sentences
        done = [False for _ in range(batch_size)]

        if self.span_aware_decoding:

            inserted_id = model_specific_kwargs["inserted_id"]
            end_of_inserted_id = model_specific_kwargs["end_of_inserted_id"]

            original_inputs = model_specific_kwargs["original_inputs"]
            oracle_token_labels = model_specific_kwargs.get("oracle_token_labels")
            oracle_mixin_p = model_specific_kwargs.get("oracle_mixin_p")

            if oracle_token_labels is not None:
                force_tokens = np.logical_not(
                    np.array(oracle_token_labels.cpu()))
                
                if oracle_mixin_p != 1.0:
                    token_classification = np.logical_not(
                        np.array(self.classify_ids_api(original_inputs).cpu()))
                    force_tokens = mixin_with_p(force_tokens,token_classification,oracle_mixin_p)
            else:
                force_tokens = np.logical_not(
                    np.array(self.classify_ids_api(original_inputs).cpu()))

            spans_to_force = []
            for i, tokens in enumerate(force_tokens):
                spans = separate_regions(original_inputs[i], tokens)
                spans_to_force.append(
                    [np.concatenate((x.cpu().numpy(), [inserted_id])) for x in spans if not all([y==1 for y in x])])

            inserted_id = model_specific_kwargs["inserted_id"]
            end_of_inserted_id = model_specific_kwargs["end_of_inserted_id"]

            # Tracks which span a beam is in
            beam_span = np.zeros((batch_size, num_beams)).astype(int)

            # Track which token a beam needs to force in each span
            beam_span_index = np.zeros((batch_size, num_beams)).astype(int)

            # Track whether a beam should be doing generating wihtout constraints
            beam_is_restricted = np.tile(force_tokens[:,1],(num_beams,1)).T.astype(int)

        while cur_len < max_length:
            model_inputs = self.prepare_inputs_for_generation(
                input_ids, past=past, attention_mask=attention_mask, use_cache=use_cache, **model_specific_kwargs
            )
            # (batch_size * num_beams, cur_len, vocab_size)
            outputs = self(**model_inputs)
            # (batch_size * num_beams, vocab_size)
            next_token_logits = outputs[0][:, -1, :]

            # if model has past, then set the past variable to speed up decoding
            if self._use_cache(outputs, use_cache):
                past = outputs[1]
            if self.config.is_encoder_decoder and do_sample is False:
                # TODO (PVP) still a bit hacky here - there might be a better solution
                next_token_logits = self.adjust_logits_during_generation(
                    next_token_logits, cur_len=cur_len, max_length=max_length
                )

            # (batch_size * num_beams, vocab_size)
            scores = F.log_softmax(next_token_logits, dim=-1)

            scores = self.postprocess_next_token_scores(
                scores=scores,
                input_ids=input_ids,
                no_repeat_ngram_size=no_repeat_ngram_size,
                bad_words_ids=bad_words_ids,
                cur_len=cur_len,
                min_length=min_length,
                max_length=max_length,
                eos_token_id=eos_token_id,
                repetition_penalty=repetition_penalty,
                batch_size=batch_size,
                num_beams=num_beams,
            )

            assert scores.shape == (batch_size * num_beams, vocab_size), "Shapes of scores: {} != {}".format(
                scores.shape, (batch_size * num_beams, vocab_size)
            )

            if do_sample:
                _scores = scores + beam_scores[:, None].expand_as(scores)  # (batch_size * num_beams, vocab_size)
                # Temperature
                if temperature != 1.0:
                    _scores = _scores / temperature
                # Top-p/top-k filtering
                _scores = top_k_top_p_filtering(
                    _scores, top_k=top_k, top_p=top_p, min_tokens_to_keep=2
                )  # (batch_size * num_beams, vocab_size)
                # re-organize to group the beam together to sample from all beam_idxs
                _scores = _scores.contiguous().view(
                    batch_size, num_beams * vocab_size
                )  # (batch_size, num_beams * vocab_size)

                # Sample 2 next tokens for each beam (so we have some spare tokens and match output of greedy beam search)
                probs = F.softmax(_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=2 * num_beams)  # (batch_size, num_beams * 2)
                # Compute next scores
                next_scores = torch.gather(_scores, -1, next_tokens)  # (batch_size, num_beams * 2)
                # sort the sampled vector to make sure that the first num_beams samples are the best
                next_scores, next_scores_indices = torch.sort(next_scores, descending=True, dim=1)
                next_tokens = torch.gather(next_tokens, -1, next_scores_indices)  # (batch_size, num_beams * 2)

            else:
                next_scores = scores + beam_scores[:, None].expand_as(scores)  # (batch_size * num_beams, vocab_size)

                # re-organize to group the beam together (we are keeping top hypothesis accross beams)
                next_scores = next_scores.view(
                    batch_size, num_beams * vocab_size
                )  # (batch_size, num_beams * vocab_size)

                next_scores, next_tokens = torch.topk(next_scores, 2 * num_beams, dim=1, largest=True, sorted=True)

            assert next_scores.size() == next_tokens.size() == (batch_size, 2 * num_beams)

            # next batch beam content
            next_batch_beam = []

            # for each sentence
            for batch_idx in range(batch_size):

                # if we are done with this sentence, add a pad token
                if done[batch_idx]:
                    assert (
                        len(generated_hyps[batch_idx]) >= num_beams
                    ), "Batch can only be done if at least {} beams have been generated".format(num_beams)
                    assert (
                        eos_token_id is not None and pad_token_id is not None
                    ), "generated beams >= num_beams -> eos_token_id and pad_token have to be defined"
                    next_batch_beam.extend([(0, pad_token_id, 0)] * num_beams)  # pad the batch
                    continue

                # next sentence beam content, this will get added to next_batch_beam
                next_sent_beam = []

            assert next_scores.size() == next_tokens.size() == (batch_size, 2 * num_beams)
            # We can modify each beam's next scores here

            # if self.span_aware_decoding and not cur_len==1:
            #     next_scores[np.where(beam_is_restricted),0] = 1
            #     # next_tokens[np.where(beam_is_restricted), 0] = spans_to_force[np.where(beam_is_restricted)]

            # next batch beam content
            next_batch_beam = []

            # for each sentence
            for batch_idx in range(batch_size):

                # if we are done with this sentence, add a pad token
                if done[batch_idx]:
                    assert (
                        len(generated_hyps[batch_idx]) >= num_beams
                    ), "Batch can only be done if at least {} beams have been generated".format(num_beams)
                    assert (
                        eos_token_id is not None and pad_token_id is not None
                    ), "generated beams >= num_beams -> eos_token_id and pad_token have to be defined"
                    next_batch_beam.extend(
                        [(0, pad_token_id, 0)] * num_beams)  # pad the batch
                    continue

                # next sentence beam content, this will get added to next_batch_beam
                next_sent_beam = []
                
                # next tokens for this sentence
                for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(
                    zip(next_tokens[batch_idx], next_scores[batch_idx])
                ):
                    # get beam and token IDs
                    beam_id = beam_token_id // vocab_size
                    token_id = beam_token_id % vocab_size

                    # Do we want to ignore the first token?
                    # if self.span_aware_decoding and cur_len>1 and not beam_took_token = [False]:
                    #     # if self.span_aware_decoding and not cur_len==1:
                    #     beam_restriced = beam_is_restricted[batch_idx, beam_id]
                    #     beam_span_id = beam_span[batch_idx, beam_id]
                    #     if beam_restriced:
                    #         span_to_force = spans_to_force[batch_idx][beam_span_id]
                    #         span_token_id = beam_span_index[batch_idx, beam_id]
                    #         token_to_force = span_to_force[span_token_id]

                    #         beam_took_token[beam_id] +=1
                    #         token_id = token_to_force
                    #         beam_token_score = torch.tensor(1)

                    #         # If we completed a span...
                    #         if span_token_id == len(span_to_force) - 1:
                    #             # Isn't reaching here..
                    #             beam_span[batch_idx, beam_id] += 1
                    #             beam_span_index[batch_idx, beam_id] = 0
                    #             beam_is_restricted[batch_idx, beam_id] = 0
                    #         beam_span_index[batch_idx, beam_id] += 1

                    #     # We've reached the end of a free span, and we have more spans to generate, so restrict on next run:
                    #     elif token_id == end_of_inserted_id and not len(spans_to_force[batch_idx][beam_span_id:]) == 0:
                    #         beam_is_restricted[batch_idx, beam_id] = 0
                    # Token is getting set correctly but it's not getting added to the beam>
                    effective_beam_id = batch_idx * num_beams + beam_id
                    # add to generated hypotheses if end of sentence
                    if (eos_token_id is not None) and (token_id.item() == eos_token_id):
                        # if beam_token does not belong to top num_beams tokens, it should not be added
                        is_beam_token_worse_than_top_num_beams = beam_token_rank >= num_beams
                        if is_beam_token_worse_than_top_num_beams:
                            continue
                        generated_hyps[batch_idx].add(
                            input_ids[effective_beam_id].clone(
                            ), beam_token_score.item(),
                        )
                    else:
                        # add next predicted token since it is not eos_token
                        next_sent_beam.append(
                            (beam_token_score, token_id, effective_beam_id))

                    # once the beam for next step is full, don't add more tokens to it.
                    if len(next_sent_beam) == num_beams:
                        break

                # Check if we are done so that we can save a pad step if all(done)
                done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                    next_scores[batch_idx].max().item(), cur_len
                )

                # update next beam content
                assert len(
                    next_sent_beam) == num_beams, "Beam should always be full"
                next_batch_beam.extend(next_sent_beam)
                assert len(next_batch_beam) == num_beams * (batch_idx +
                                                            1), "We should have added num_beams each step"

            # stop when we are done with each sentence
            if all(done):
                break

            # sanity check / prepare next batch
            # For whatever reason it feels like the forced token isn't ending up in next_batch_beam
          

            #Force tokens if necessary
            if self.span_aware_decoding:
                span_aware_next_beam_batch = []
                for beam_score, beam_token, effective_beam_id in next_batch_beam:
                    beam_id = effective_beam_id % num_beams
                    batch_idx = (effective_beam_id - beam_id) // num_beams
                    

                    beam_restriced = beam_is_restricted[batch_idx, beam_id]
                    beam_span_id = beam_span[batch_idx, beam_id]
                    
                    #Beam is restriced and there's more work to do
                    if beam_restriced and beam_span_id < len(spans_to_force[batch_idx]):
                        span_to_force = spans_to_force[batch_idx][beam_span_id]
                        span_token_id = beam_span_index[batch_idx, beam_id]
                        token_to_force = span_to_force[span_token_id]
                         
                        token_id = token_to_force


                        # If we completed a span...
                        if span_token_id == len(span_to_force) - 1:
                            # Isn't reaching here..
                            beam_span[batch_idx, beam_id] += 1
                            beam_span_index[batch_idx, beam_id] = 0
                            beam_is_restricted[batch_idx, beam_id] = 0
                        else:
                            beam_span_index[batch_idx, beam_id] += 1
                        span_aware_next_beam_batch.append((torch.tensor(0), token_to_force, effective_beam_id))
                    else:
                        span_aware_next_beam_batch.append((beam_score, beam_token, effective_beam_id))
                        if beam_token == end_of_inserted_id:
                            #Start forcing on the next loop:
                            beam_is_restricted[batch_idx, beam_id] = 1

                next_batch_beam = span_aware_next_beam_batch

            assert len(next_batch_beam) == batch_size * num_beams
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_tokens = input_ids.new([x[1] for x in next_batch_beam])
            beam_idx = input_ids.new(
                [x[2] for x in next_batch_beam])  # Effective beam ids
            
            # re-order batch and update current length
            input_ids = input_ids[beam_idx, :]
            input_ids = torch.cat(
                [input_ids, beam_tokens.unsqueeze(1)], dim=-1)

            # if cur_len > 1 and self.span_aware_decoding:
            #     # Not doing this in numpy because it's 1am
            #     for i in range(len(input_ids)):
            #         beam_id = i % num_beams
            #         batch_idx = i // num_beams

            #         beam_restriced = beam_is_restricted[batch_idx, beam_id]
            #         if beam_restriced:
            #             span_to_force = spans_to_force[batch_idx][beam_span[batch_idx, beam_id]]
            #             span_token_id = beam_span_index[batch_idx, beam_id]
            #             token_to_force = span_to_force[span_token_id]
            #             if i == 5:
            #                 a = 1
            #             input_ids[i, -1] = int(token_to_force)
            #             # beam_token_score = torch.tensor(1)

            #             # If we completed a span...
            #             if span_token_id == len(span_to_force) - 1:
            #                 # Isn't reaching here..
            #                 beam_span[batch_idx, beam_id] += 1
            #                 beam_span_index[batch_idx, beam_id] = 0
            #                 beam_is_restricted[batch_idx, beam_id] = 0
            #             else:
            #                 beam_span_index[batch_idx, beam_id] += 1

            #         # We've reached the end of a free span, restrict on next run:
            #         elif token_id == end_of_inserted_id:
            #             beam_is_restricted[batch_idx, beam_id] = 0
            # Values aren't getting set correctly
            cur_len = cur_len + 1

            # re-order internal states
            if past is not None:
                past = self._reorder_cache(past, beam_idx)

            # extend attention_mask for new generated input if only decoder
            if self.config.is_encoder_decoder is False:
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )

        # finalize all open beam hypotheses and add to generated hypotheses
        for batch_idx in range(batch_size):
            if done[batch_idx]:
                continue

            # test that beam scores match previously calculated scores if not eos and batch_idx not done
            if eos_token_id is not None and all(
                (token_id % vocab_size).item() != eos_token_id for token_id in next_tokens[batch_idx]
            ):
                assert torch.all(
                    next_scores[batch_idx, :num_beams] == beam_scores.view(batch_size, num_beams)[
                        batch_idx]
                ), "If batch_idx is not done, final next scores: {} have to equal to accumulated beam_scores: {}".format(
                    next_scores[:, :num_beams][batch_idx], beam_scores.view(batch_size, num_beams)[
                        batch_idx],
                )

            # need to add best num_beams hypotheses to generated hyps
            for beam_id in range(num_beams):
                effective_beam_id = batch_idx * num_beams + beam_id
                final_score = beam_scores[effective_beam_id].item()
                final_tokens = input_ids[effective_beam_id]
                generated_hyps[batch_idx].add(final_tokens, final_score)

        # depending on whether greedy generation is wanted or not define different output_batch_size and output_num_return_sequences_per_batch
        output_batch_size = batch_size if do_sample else batch_size * num_return_sequences
        output_num_return_sequences_per_batch = 1 if do_sample else num_return_sequences

        # select the best hypotheses
        sent_lengths = input_ids.new(output_batch_size)
        best = []

        # retrieve best hypotheses
        for i, hypotheses in enumerate(generated_hyps):
            sorted_hyps = sorted(hypotheses.beams, key=lambda x: x[0])
            for j in range(output_num_return_sequences_per_batch):
                effective_batch_idx = output_num_return_sequences_per_batch * i + j
                best_hyp = sorted_hyps.pop()[1]
                sent_lengths[effective_batch_idx] = len(best_hyp)
                best.append(best_hyp)

        # shorter batches are padded
        if sent_lengths.min().item() != sent_lengths.max().item():
            assert pad_token_id is not None, "`Pad_token_id` has to be defined"
            sent_max_len = min(sent_lengths.max().item() + 1, max_length)
            decoded = input_ids.new(
                output_batch_size, sent_max_len).fill_(pad_token_id)

            # fill with hypothesis and eos_token_id if necessary
            for i, hypo in enumerate(best):
                decoded[i, : sent_lengths[i]] = hypo
                if sent_lengths[i] < max_length:
                    decoded[i, sent_lengths[i]] = eos_token_id
        else:
            # none of the hypotheses have an eos_token
            assert (len(hypo) == max_length for hypo in best)
            decoded = torch.stack(best).type(torch.long).to(
                next(self.parameters()).device)

        return decoded


class HyperbolicLibraryEmbedding():

    def __init__(self, tree_embedding_path, tokenizer):
        self.tree_embedding_path = tree_embedding_path
        self.tree = load_pickled_tree(self.tree_embedding_path)
        self.num_nodes = self.tree.number_of_nodes()
        self.device = "cpu"
        self.model = self.load_model()
        self.tokenizer = tokenizer
        self.node_embeddings = self.extract_embedding()

        # Since we don't always id,
        self.token_embeddings = defaultdict(list)
        for node_id, data in list(self.tree.nodes(data=True)):
            euclidean_component = [
                x for x in self.node_embeddings[node_id] if x["space"] == "euclidean"][0]

            node_name_input_ids = self.tokenizer(
                data["name"], add_special_tokens=False)["input_ids"]
            for input_id in node_name_input_ids:
                self.token_embeddings[input_id].append(
                    euclidean_component["vector"])

        self.average_token_embeddings = {k: np.mean(
            v, axis=0) for k, v in self.token_embeddings.items()}

    def get_last_checkpoint_path(self):
        candidate_paths = [x for x in pathlib.Path(
            self.tree_embedding_path).glob('**/*embeddings*') if x.is_file()]

        def get_checkpoint_epoch(x): return x.name.split(".")[-1]
        epochs = [get_checkpoint_epoch(x) for x in candidate_paths]

        if "final" in epochs:
            return candidate_paths[epochs.index("final")]
        else:
            return candidate_paths[np.argmax([int(x) for x in epochs])]

    def load_model(self):
        last_checkpoint_path = self.get_last_checkpoint_path()
        logger.info(f"Loading tree embedding from {last_checkpoint_path}")
        return torch.load(last_checkpoint_path).to(self.device)

    def set_embeddings(self, model):
        embeddings = model.get_input_embeddings()
        for k, v in self.average_token_embeddings.items():
            assert v.shape[0] == embeddings.embedding_dim
            embeddings.weight.data[k] = torch.tensor(
                v).to(embeddings.weight.data.device)
        model.set_input_embeddings(embeddings)

    def extract_embedding(self):
        embedding = dict()
        for node in range(self.num_nodes):
            embedding[node] = []
            for hyp_factor in self.model.H:
                embedding[node].append(
                    {'space': 'hyperbolic',
                     'scale': hyp_factor.scale().item(),
                     'vector': hyp_factor.w[node, :].cpu().detach().numpy()
                     }
                )
            for sph_factor in self.model.S:
                embedding[node].append(
                    {'space': 'spherical',
                     'scale': sph_factor.scale().item(),
                     'vector': sph_factor.w[node, :].cpu().detach().numpy()
                     }
                )
            if len(self.model.E) > 0:
                embedding[node].append(
                    {'space': 'euclidean',
                     'scale': 0,
                     'vector': self.model.E[0].w[node, :].cpu().detach().numpy()
                     }
                )
        return embedding
