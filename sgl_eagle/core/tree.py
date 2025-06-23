import torch
from transformers.cache_utils import DynamicCache

def tree_decoding(
        draft_model,
        tree_candidates,
        past_key_values,
        tree_position_ids,
        input_ids,
        retrieve_indices,
):
    """
    Perform the tree-style decoding from candidates organized in a tree structure.

    Args:
        draft_model: The model to use for decoding.
        tree_candidates: The candidates to use for decoding.
        past_key_values: The past key values to use for decoding.
        tree_position_ids: The tree position ids to use for decoding.
    """
    position_ids = tree_position_ids + input_ids.shape[1]
    if position_ids is not None and position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)
    outputs, tree_logits, hidden_state = model(
        tree_candidates,
        output_orig=True,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )

    hidden_state = torch.cat(outputs["hidden_states"], dim=-1)
    logits = tree_logits[0, retrieve_indices]
    return logits, hidden_state, outputs


@torch.no_grad()
def update_inference_inputs(
        input_ids,
        candidates,
        best_candidate,
        accept_length,
        retrieve_indices,
        logits_processor,
        new_token,
        past_key_values_data_list,
        current_length_data,
        model,
        hidden_state_new,
        sample_p
):
    prev_input_len = input_ids.shape[1]
    # Map the best candidate indices to the original indices in the sequence
    select_indices = (
            retrieve_indices[best_candidate, : accept_length + 1] + prev_input_len
    )
    # Append the tokens from the best candidate to the input sequence
    input_ids = torch.cat(
        [input_ids, candidates[None, best_candidate, : accept_length + 1].to(input_ids.device)], dim=-1
    )
    # Update the past key values based on the selected tokens
    # Source tensor that contains relevant past information based on the selected candidate
    for past_key_values_data in past_key_values_data_list:
        tgt = past_key_values_data[..., select_indices.to(past_key_values_data.device), :]
        # Destination tensor where the relevant past information will be stored
        dst = past_key_values_data[..., prev_input_len: prev_input_len + tgt.shape[-2], :]
        # Copy relevant past information from the source to the destination
        dst.copy_(tgt, non_blocking=True)

    # Update the current length tensor (currently only support batch size is 1)
    current_length_data.fill_(prev_input_len + tgt.shape[-2])

    retrieve_hidden_state_new = hidden_state_new[:, retrieve_indices]
    accept_hidden_state_new = retrieve_hidden_state_new[:, best_candidate, : accept_length + 1]
    # token=model.base_model.lm_head(accept_hidden_state_new[:,-1]).argmax()
    # token=token[None,None]
    prob = sample_p
    if logits_processor is not None:
        token = torch.multinomial(prob, 1)
        token = token[None]
    else:
        token = torch.argmax(prob)
        token = token[None, None]
    # hidden_state = torch.cat((hidden_state, accept_hidden_state_new), dim=1)
    draft_tokens, retrieve_indices,tree_mask,tree_position_ids = model.ea_layer.topK_genrate(accept_hidden_state_new,
                                              input_ids=torch.cat((input_ids, token.to(input_ids.device)), dim=1),
                                              head=model.base_model.lm_head,logits_processor=logits_processor)


    new_token += accept_length + 1

    return input_ids, draft_tokens, retrieve_indices,tree_mask,tree_position_ids, new_token, None, token




def generate_topk(
        hidden_states, 
        input_ids, 
        draft_model, 
        logits_processor,
        num_spec_steps, 
        num_draft_tokens):
    sample_token = input_ids[:, -1]

    scores_list = []
    parents_list = []
    ss_token = []

    input_ids = input_ids[:, 1:]

    len_posi = input_ids.shape[1]

    if hasattr(draft_model, "stable_kv") and draft_model.stable_kv is not None:
        kv_len = draft_model.stable_kv[0][0].shape[2]
        out_hidden, past_key_values = draft_model(
            hidden_states, 
            input_ids=input_ids[:, kv_len:],
            past_key_values=draft_model.stable_kv, 
            use_cache=True,
        )
    else:
        past_key_values = DynamicCache()
        out_hidden, past_key_values = draft_model(
            hidden_states, 
            input_ids=input_ids, 
            past_key_values=past_key_values, 
            use_cache=True)
        draft_model.stable_kv = past_key_values

    print(out_hidden.shape)
    exit()

    last_hidden = out_hidden[:, -1]
    last_headout = self.lm_head(self.norm(last_hidden))

    last_p = self.logsoftmax(last_headout)
    top = torch.topk(last_p, top_k, dim=-1)
    topk_index, topk_p = top.indices, top.values
    scores = topk_p[0]
    scores_list.append(scores[None])
    parents_list.append(torch.zeros(1, dtype=torch.long, device=scores.device))
    if self.config.vocab_size==self.config.draft_vocab_size:
        ss_token.append(topk_index)
        input_ids = topk_index
    else:
        ss_token.append(topk_index+self.d2t[topk_index])
        input_ids = topk_index+self.d2t[topk_index]
    input_hidden = last_hidden[None].repeat(1, top_k, 1)
    tree_mask = self.tree_mask_init
    topk_cs_index = torch.arange(top_k, device=self.embed_tokens.weight.device)

    # 4
    for i in range(depth):
        self.tree_mask = tree_mask
        position_ids = len_posi + self.position_ids
        # with Timer("draft one"):
        out_hidden, past_key_values = self(input_hidden, input_ids=input_ids, past_key_values=past_key_values,
                                            position_ids=position_ids, use_cache=True)
        len_posi += 1

        # with Timer("sort1"):
        bias1 = top_k if i > 0 else 0
        bias2 = max(0, i - 1)
        bias = 1 + top_k ** 2 * bias2 + bias1
        parents = (topk_cs_index + bias)
        parents_list.append(parents)

        last_headout = self.lm_head(self.norm(out_hidden[0]))
        last_p = self.logsoftmax(last_headout)

        top = torch.topk(last_p, top_k, dim=-1)
        topk_index, topk_p = top.indices, top.values

        cu_scores = topk_p + scores[:, None]

        topk_cs = torch.topk(cu_scores.view(-1), top_k, dim=-1)
        topk_cs_index, topk_cs_p = topk_cs.indices, topk_cs.values
        scores = topk_cs_p

        out_ids = topk_cs_index // top_k
        input_hidden = out_hidden[:, out_ids]

        input_ids = topk_index.view(-1)[topk_cs_index][None]

        if self.config.vocab_size == self.config.draft_vocab_size:
            ss_token.append(topk_index)
        else:
            input_ids = input_ids + self.d2t[input_ids]
            ss_token.append(topk_index+self.d2t[topk_index])
        scores_list.append(cu_scores)
        tree_mask = torch.cat((tree_mask[:, :, out_ids], self.tree_mask_init), dim=3)


    scores_list = torch.cat(scores_list, dim=0).view(-1)
    ss_token_list = torch.cat(ss_token, dim=0).view(-1)
    top_scores = torch.topk(scores_list, total_tokens, dim=-1)
    top_scores_index = top_scores.indices
    top_scores_index = torch.sort(top_scores_index).values

    draft_tokens = ss_token_list[top_scores_index]
    draft_tokens = torch.cat((sample_token, draft_tokens), dim=0)

    draft_parents = torch.cat(parents_list, dim=0)[top_scores_index // top_k].long()
    mask_index = torch.searchsorted(top_scores_index, draft_parents - 1, right=False)
    # mask_index[(top_scores_index[mask_index]!=draft_parents - 1)]=-1
    mask_index[draft_parents == 0] = -1
    mask_index = mask_index + 1
    mask_index_list = mask_index.tolist()
    # with Timer("mask"):
    tree_mask = torch.eye(total_tokens + 1).bool()
    tree_mask[:, 0] = True
    for i in range(total_tokens):
        tree_mask[i + 1].add_(tree_mask[mask_index_list[i]])


    tree_position_ids = torch.sum(tree_mask, dim=1) - 1

    tree_mask = tree_mask.float()[None, None]
    draft_tokens = draft_tokens[None]

    del parents_list, scores_list, ss_token, ss_token_list, draft_parents

    # with Timer("retrieve"):

    max_depth = torch.max(tree_position_ids) + 1
    noleaf_index = torch.unique(mask_index).tolist()
    noleaf_num = len(noleaf_index) - 1
    leaf_num = total_tokens - noleaf_num

    retrieve_indices = torch.zeros(leaf_num, max_depth.item(), dtype=torch.long) - 1
    retrieve_indices = retrieve_indices.tolist()

    rid = 0
    position_ids_list = tree_position_ids.tolist()

    for i in range(total_tokens + 1):
        if i not in noleaf_index:
            cid = i
            depth = position_ids_list[i]
            for j in reversed(range(depth + 1)):
                retrieve_indices[rid][j] = cid
                cid = mask_index_list[cid - 1]
            rid += 1

    if logits_processor is not None:
        maxitem = total_tokens + 5

        def custom_sort(lst):
            # sort_keys=[len(list)]
            sort_keys = []
            for i in range(len(lst)):
                sort_keys.append(lst[i] if lst[i] >= 0 else maxitem)
            return sort_keys

        retrieve_indices = sorted(retrieve_indices, key=custom_sort)

    retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
    del mask_index, mask_index_list, noleaf_index, noleaf_num, leaf_num, max_depth, rid
    tree_position_ids = tree_position_ids.to(hidden_states.device)

    return draft_tokens, retrieve_indices, tree_mask, tree_position_ids



def initialize_tree(
        input_ids, 
        base_model, 
        draft_model, 
        past_key_values, 
        logits_processor,
        num_spec_steps, 
        num_draft_tokens):
    """
    Initialize the tree for the first token.

    Args:
        input_ids: The input ids.
        base_model: The base model.
        draft_model: The draft model.
        past_key_values: The past key values.
        logits_processor: The logits processor.

    """
    # perform a forward pass and get logits and hidden states
    outputs = base_model(input_ids, past_key_values=past_key_values, output_hidden_states=True)
    logits = outputs.logits

    if logits_processor is not None:
        logits = logits[:, -1]
        logits = logits_processor(None, logits)
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        token = torch.multinomial(probabilities, 1)
    else:
        token = torch.argmax(logits[:, -1])
        token = token[None, None]

    # this is the input IDs to draft model
    input_ids = torch.cat((input_ids, token.to(input_ids.device)), dim=1)
    hidden_states=torch.cat(outputs["hidden_states"],dim=-1)

    draft_tokens, retrieve_indices,tree_mask,tree_position_ids = generate_topk(
        hidden_states=hidden_states,
        input_ids=input_ids,
        draft_model=draft_model,
        logits_processor=logits_processor,
        num_spec_steps=num_spec_steps,
        num_draft_tokens=num_draft_tokens,
    )
    return draft_tokens, retrieve_indices,tree_mask,tree_position_ids, orig, hidden_states, token