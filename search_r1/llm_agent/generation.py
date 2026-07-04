import torch
import re
from collections import defaultdict
import os
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from .tensor_helper import TensorHelper, TensorConfig
from verl import DataProto
from verl.utils.tracking import Tracking
import shutil
import requests
from search_r1.budgeting import (
    build_budget_mask,
    parse_budget_declaration,
    should_force_search,
)

@dataclass
class GenerationConfig:
    max_turns: int
    max_start_length: int
    max_prompt_length: int 
    max_response_length: int
    max_obs_length: int
    num_gpus: int
    no_think_rl: bool=False
    search_url: str = None
    topk: int = 3
    retriever_name: str = "e5"
    index_path: str = None
    corpus_path: str = None
    retriever_model: str = "intfloat/e5-base-v2"
    faiss_gpu: bool = True
    retrieval_batch_size: int = 512
    faiss_nprobe: int = None
    enable_budget_planner: bool = False
    max_budget: int = 5
    min_searches: int = 0

class LLMGenerationManager:
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: GenerationConfig,
        is_validation: bool = False,
    ):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation
        self.local_retriever = None

        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=config.max_obs_length,
            max_start_length=config.max_start_length
        ))

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids']

    def _postprocess_responses(self, responses: torch.Tensor) -> torch.Tensor:
        """Process responses to stop at search operation or answer operation."""
        responses_str = self.tokenizer.batch_decode(
            responses, 
            skip_special_tokens=True
        )

        stop_tags = ['</search>', '</answer>']
        if self.config.enable_budget_planner:
            stop_tags.append('</budget>')

        truncated_responses = []
        for resp in responses_str:
            stop_positions = [(resp.find(tag) + len(tag), tag) for tag in stop_tags if tag in resp]
            if stop_positions:
                stop_pos, _ = min(stop_positions, key=lambda item: item[0])
                truncated_responses.append(resp[:stop_pos])
            else:
                truncated_responses.append(resp)
        responses_str = truncated_responses

        if self.config.no_think_rl:
            raise ValueError('stop')
            # if no_think_rl is enabled, only keep action in the str
            actions, _ = self.env.postprocess_predictions(responses_str)
            responses_str=[f"<answer>{envs[idx].ACTION_LOOKUP[action]}</answer>" for idx, action in enumerate(actions)]
            print("RESPONSES:", responses_str)
        responses = self._batch_tokenize(responses_str)
        return responses, responses_str

    def _process_next_obs(self, next_obs: List[str]) -> torch.Tensor:
        """Process next observations from environment."""
        
        next_obs_ids = self.tokenizer(
            next_obs, 
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,  # Prevents adding special tokens
        )['input_ids']

        if next_obs_ids.shape[1] > self.config.max_obs_length:
            print(f"[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, {next_obs_ids.shape[1]} & {self.config.max_obs_length}")            
            next_obs_ids = next_obs_ids[:, :self.config.max_obs_length]

        return next_obs_ids

    def _update_rolling_state(self, rollings: DataProto, cur_responses: torch.Tensor, 
                            next_obs_ids: torch.Tensor) -> Dict:
        """Update rolling state with new responses and observations."""
        # Concatenate and handle padding        
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            cur_responses,
            next_obs_ids
        ])
        
        # Create attention mask and position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)

        new_rollings = DataProto.from_dict({
            'input_ids': new_input_ids[:, -max_len:],
            'position_ids': new_position_ids[:, -max_len:],
            'attention_mask': new_attention_mask[:, -max_len:]
        })
        new_rollings.meta_info.update(rollings.meta_info)
        
        return new_rollings

    def _info_masked_concatenate_with_padding(self, 
                prompt: torch.Tensor, 
                prompt_with_mask: torch.Tensor, 
                response: torch.Tensor, 
                info: torch.Tensor = None,
                pad_to_left: bool = True
            ) -> torch.Tensor:
        """Concatenate tensors and handle padding. Additionally, create a mask (info_mask) to cover the information block if it exists."""
        pad_id = self.tokenizer.pad_token_id
        tensors = [prompt, response]
        tensors_with_mask = [prompt_with_mask, response]
        if info is not None:
            tensors.append(info)
            info_mask = torch.full(info.size(), pad_id, dtype=info.dtype, device=info.device) # information mask
            tensors_with_mask.append(info_mask)
        
        concatenated = torch.cat(tensors, dim=1)
        concatenated_with_info = torch.cat(tensors_with_mask, dim=1)
        mask = concatenated != pad_id if pad_to_left else concatenated == pad_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        padded_tensor = concatenated.gather(1, sorted_indices)
        padded_tensor_with_info = concatenated_with_info.gather(1, sorted_indices)

        return padded_tensor, padded_tensor_with_info

    def _update_right_side(self, right_side: Dict, 
                          cur_responses: torch.Tensor,
                          next_obs_ids: torch.Tensor = None) -> Dict:
        """Update right side state."""
        if next_obs_ids != None:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    next_obs_ids, 
                    pad_to_left=False
                )
        else:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    pad_to_left=False
                )
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        return {'responses': responses[:, :max_len], 'responses_with_info_mask': responses_with_info_mask[:, :max_len]}

    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """
            Wrapper for generation that handles multi-GPU padding requirements.
            if num_gpus <= 1, return self.actor_rollout_wg.generate_sequences(active_batch)
            if active_batch size is not divisible by num_gpus, pad with first sequence
            then remove padding from output
        """
        num_gpus = self.config.num_gpus
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)
            
        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus
        
        for key in active_batch.batch.keys():
            active_batch.batch[key] = active_batch.batch[key].long()
        if remainder == 0:
            return self.actor_rollout_wg.generate_sequences(active_batch)
        
        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}
        
        for k, v in active_batch.batch.items():
            # Use first sequence as padding template
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)

        padded_active_batch = DataProto.from_dict(padded_batch)
        for key in padded_active_batch.batch.keys():
            padded_active_batch.batch[key] = padded_active_batch.batch[key].long()

        # Generate with padded batch
        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)

        # Remove padding from output
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        
        # Handle meta_info if present
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta
            
        padded_output.batch = trimmed_batch
        return padded_output

    def run_llm_loop(self, gen_batch, initial_input_ids: torch.Tensor,
                     force_search_active: bool = False) -> Tuple[Dict, Dict]:
        """Run main LLM generation loop.

        force_search_active is the train-only forced-execution switch,
        eval leaves it False so adaptivity is measured unforced.
        """
        
        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        original_right_side = {'responses': initial_input_ids[:, []], 'responses_with_info_mask': initial_input_ids[:, []]}
        
        active_mask = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.bool)
        turns_stats = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        valid_action_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        valid_search_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        declared_budgets = [-1 for _ in range(gen_batch.batch['input_ids'].shape[0])]
        search_counts = [0 for _ in range(gen_batch.batch['input_ids'].shape[0])]
        blocked_search_counts = [0 for _ in range(gen_batch.batch['input_ids'].shape[0])]
        active_num_list = [active_mask.sum().item()]
        rollings = gen_batch

        # Main generation loop
        for step in range(self.config.max_turns):
            if not active_mask.sum():
                break
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            
            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })            
            # re-attach meta_info from_dict drops for do_sample
            rollings_active.meta_info = rollings.meta_info
            gen_output = self._generate_with_gpu_padding(rollings_active)

            meta_info = gen_output.meta_info            
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            # Execute in environment and process observations
            next_obs, dones, valid_action, is_search = self.execute_predictions(
                responses_str,
                self.tokenizer.pad_token,
                active_mask,
                declared_budgets=declared_budgets,
                search_counts=search_counts,
                blocked_search_counts=blocked_search_counts,
                force_search_active=force_search_active,
            )
            
            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            turns_stats[curr_active_mask] += 1
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)

            next_obs_ids = self._process_next_obs(next_obs)
            
            # Update states
            rollings = self._update_rolling_state(
                rollings,
                responses_ids,
                next_obs_ids
            )
            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
                next_obs_ids
            )
            
        # final LLM rollout
        if active_mask.sum():
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )

            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })            
            # re-attach meta_info from_dict drops for do_sample
            rollings_active.meta_info = rollings.meta_info
            gen_output = self._generate_with_gpu_padding(rollings_active)

            meta_info = gen_output.meta_info            
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            # # Execute in environment and process observations
            _, dones, valid_action, is_search = self.execute_predictions(
                responses_str,
                self.tokenizer.pad_token,
                active_mask,
                do_search=False,
                declared_budgets=declared_budgets,
                search_counts=search_counts,
                blocked_search_counts=blocked_search_counts,
            )

            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)
            

            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
            )
        
        meta_info['turns_stats'] = turns_stats.tolist()
        meta_info['active_mask'] = active_mask.tolist()
        meta_info['valid_action_stats'] = valid_action_stats.tolist()
        meta_info['valid_search_stats'] = valid_search_stats.tolist()
        meta_info['declared_budget_stats'] = declared_budgets
        meta_info['blocked_search_stats'] = blocked_search_counts
        
        print("ACTIVE_TRAJ_NUM:", active_num_list)
        
        rollout_stats = {
            'declared_budget': torch.tensor(declared_budgets, dtype=torch.long),
            'valid_search_count': torch.tensor(search_counts, dtype=torch.long),
            'blocked_search_count': torch.tensor(blocked_search_counts, dtype=torch.long),
        }

        return self._compose_final_output(original_left_side, original_right_side, meta_info, rollout_stats)

    def _compose_final_output(self, left_side: Dict,
                            right_side: Dict,
                            meta_info: Dict,
                            rollout_stats: Dict = None) -> Tuple[Dict, Dict]:
        """Compose final generation output."""
        final_output = right_side.copy()
        final_output['prompts'] = left_side['input_ids']
        
        # Combine input IDs
        final_output['input_ids'] = torch.cat([
            left_side['input_ids'],
            right_side['responses']
        ], dim=1)
        
        # Create attention mask and position ids
        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses'])
        ], dim=1)
        final_output['info_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses_with_info_mask'])
        ], dim=1)
        
        final_output['position_ids'] = self.tensor_fn.create_position_ids(
            final_output['attention_mask']
        )

        if rollout_stats is not None:
            final_output.update(rollout_stats)

        # budget_mask rides the DataProto tensor to survive the reorder
        if self.config.enable_budget_planner and rollout_stats is not None:
            final_output['budget_mask'] = self._build_budget_mask_tensor(
                left_side['input_ids'],
                final_output['responses'],
                rollout_stats['declared_budget'],
            )

        final_output = DataProto.from_dict(final_output)
        final_output.meta_info.update(meta_info)

        return final_output

    def _build_budget_mask_tensor(self, prompt_ids: torch.Tensor,
                                  responses: torch.Tensor,
                                  declared_budgets: torch.Tensor) -> torch.Tensor:
        """Return a full-seq 0/1 mask over each row's budget declaration.

        Prompt span is all zeros, the declaration lives in the response.
        """
        prompt_mask = torch.zeros_like(prompt_ids)
        response_mask = torch.zeros_like(responses)
        for i in range(responses.size(0)):
            k = int(declared_budgets[i].item())
            if k < 0:
                continue
            budget_ids = self.tokenizer(
                f'<budget>{k}</budget>', add_special_tokens=False
            )['input_ids']
            row_mask = build_budget_mask(responses[i].tolist(), budget_ids)
            response_mask[i] = torch.tensor(
                row_mask, dtype=responses.dtype, device=responses.device
            )
        return torch.cat([prompt_mask, response_mask], dim=1)

    def execute_predictions(
        self,
        predictions: List[str],
        pad_token: str,
        active_mask=None,
        do_search=True,
        declared_budgets=None,
        search_counts=None,
        blocked_search_counts=None,
        force_search_active=False,
    ) -> List[str]:
        """
        Execute predictions across multiple environments.
        NOTE: the function is the actual `step` function in the environment
        NOTE penalty_for_invalid is not included in observation shown to the LLM
        
        Args:
            envs: List of environment instances
            predictions: List of action predictions
            pad_token: Token to use for padding
            
        Returns:
            List of observation strings
        """
        cur_actions, contents = self.postprocess_predictions(predictions)
        next_obs, dones, valid_action, is_search = [], [], [], []
        
        if declared_budgets is None:
            declared_budgets = [-1 for _ in predictions]
        if search_counts is None:
            search_counts = [0 for _ in predictions]
        if blocked_search_counts is None:
            blocked_search_counts = [0 for _ in predictions]

        def _effective_cap(idx):
            # forcing raises the cap to min_searches, else declared
            declared = declared_budgets[idx]
            if force_search_active and declared >= 0:
                return max(declared, self.config.min_searches)
            return declared

        search_queries = []
        if do_search:
            for i, (action, content) in enumerate(zip(cur_actions, contents)):
                if action != 'search':
                    continue
                if self.config.enable_budget_planner and declared_budgets[i] >= 0 and search_counts[i] >= _effective_cap(i):
                    continue
                if self.config.enable_budget_planner and declared_budgets[i] < 0:
                    continue
                search_queries.append(content)
        if do_search:
            search_results = self.batch_search(search_queries)
            assert len(search_results) == len(search_queries)
        else:
            search_results = []

        for i, (action, active) in enumerate(zip(cur_actions, active_mask)):
            
            if not active:
                next_obs.append('')
                dones.append(1)
                valid_action.append(0)
                is_search.append(0)
            else:
                if action == 'budget':
                    if not self.config.enable_budget_planner:
                        next_obs.append('\nBudget declarations are not enabled for this run. Continue with <think>, <search>, or <answer>.\n')
                        dones.append(0)
                        valid_action.append(0)
                        is_search.append(0)
                    elif declared_budgets[i] >= 0:
                        next_obs.append('\nA retrieval budget was already declared. Continue reasoning, searching, or answer now.\n')
                        dones.append(0)
                        valid_action.append(0)
                        is_search.append(0)
                    else:
                        budget = parse_budget_declaration(f'<budget>{contents[i]}</budget>', max_budget=self.config.max_budget)
                        if budget is None:
                            next_obs.append(f'\nThe budget must be an integer from 0 to {self.config.max_budget} inside <budget> and </budget>. Let me try again.\n')
                            dones.append(0)
                            valid_action.append(0)
                            is_search.append(0)
                        else:
                            declared_budgets[i] = budget
                            next_obs.append('\n')
                            dones.append(0)
                            valid_action.append(1)
                            is_search.append(0)
                elif self.config.enable_budget_planner and declared_budgets[i] < 0:
                    next_obs.append(f'\nBefore searching or answering, I must declare a retrieval budget from 0 to {self.config.max_budget} using <budget>k</budget>.\n')
                    dones.append(0)
                    valid_action.append(0)
                    is_search.append(0)
                elif action == 'answer':
                    if (
                        do_search
                        and self.config.enable_budget_planner
                        and should_force_search(
                            declared_budgets[i],
                            search_counts[i],
                            force_search_active,
                            self.config.min_searches,
                        )
                    ):
                        # bootstrap, block early answer so k binds
                        next_obs.append(
                            '\nThe declared retrieval budget has not been '
                            'used yet. I should search before answering.\n'
                        )
                        dones.append(0)
                        valid_action.append(1)
                        is_search.append(0)
                    else:
                        next_obs.append('')
                        dones.append(1)
                        valid_action.append(1)
                        is_search.append(0)
                elif action == 'search':
                    if not do_search:
                        next_obs.append('')
                        dones.append(0)
                        valid_action.append(0)
                        is_search.append(0)
                    elif self.config.enable_budget_planner and search_counts[i] >= _effective_cap(i):
                        blocked_search_counts[i] += 1
                        next_obs.append('\nThe declared retrieval budget is exhausted. I must answer using the evidence already available.\n')
                        dones.append(0)
                        valid_action.append(0)
                        is_search.append(0)
                    else:
                        search_counts[i] += 1
                        next_obs.append(f'\n\n<information>{search_results.pop(0).strip()}</information>\n\n')
                        dones.append(0)
                        valid_action.append(1)
                        is_search.append(1)
                else:
                    next_obs.append(f'\nMy previous action is invalid. \
If I want to search, I should put the query between <search> and </search>. \
If I want to give the final answer, I should put the answer between <answer> and </answer>. Let me try again.\n')
                    dones.append(0)
                    valid_action.append(0)
                    is_search.append(0)
            
        assert len(search_results) == 0
            
        return next_obs, dones, valid_action, is_search

    def postprocess_predictions(self, predictions: List[Any]) -> Tuple[List[int], List[bool]]:
        """
        Process (text-based) predictions from llm into actions and validity flags.
        
        Args:
            predictions: List of raw predictions
            
        Returns:
            Tuple of (actions list, validity flags list)
        """
        actions = []
        contents = []
                
        for prediction in predictions:
            if isinstance(prediction, str): # for llm output
                pattern = r'<(budget|search|answer)>(.*?)</\1>'
                match = re.search(pattern, prediction, re.DOTALL)
                if match:
                    content = match.group(2).strip()  # Return only the content inside the tags
                    action = match.group(1)
                else:
                    content = ''
                    action = None
            else:
                raise ValueError(f"Invalid prediction type: {type(prediction)}")
            
            actions.append(action)
            contents.append(content)
            
        return actions, contents

    def batch_search(self, queries: List[str] = None) -> str:
        """
        Batchified search for queries.
        Args:
            queries: queries to call the search engine
        Returns:
            search results which is concatenated into a string
        """
        results = self._batch_search(queries)['result']
        
        return [self._passages2string(result) for result in results]

    def _batch_search(self, queries):
        if self.config.search_url == "local":
            if self.local_retriever is None:
                self.local_retriever = self._build_local_retriever()
            results, scores = self.local_retriever.batch_search(
                query_list=queries,
                num=self.config.topk,
                return_score=True,
            )
            return {
                "result": [
                    [{"document": doc, "score": score} for doc, score in zip(single_result, single_scores)]
                    for single_result, single_scores in zip(results, scores)
                ]
            }
        
        payload = {
            "queries": queries,
            "topk": self.config.topk,
            "return_scores": True
        }
        
        return requests.post(self.config.search_url, json=payload).json()

    def _build_local_retriever(self):
        from search_r1.search.retrieval_server import Config, get_retriever

        retriever_config = Config(
            retrieval_method=self.config.retriever_name,
            retrieval_topk=self.config.topk,
            index_path=self.config.index_path,
            corpus_path=self.config.corpus_path,
            faiss_gpu=self.config.faiss_gpu,
            retrieval_model_path=self.config.retriever_model,
            retrieval_pooling_method="mean",
            # cap query encode at 64, untrained policy runs long
            retrieval_query_max_length=64,
            retrieval_use_fp16=True,
            retrieval_batch_size=self.config.retrieval_batch_size,
        )
        return get_retriever(retriever_config)

    def _passages2string(self, retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
            
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"

        return format_reference
