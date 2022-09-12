import copy
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from utils.utils import *
from utils.query_interface import KB_query, KB_query_with_timeout
from rules.grammar import V_CLASS_IDS, E_CLASS_IDS, AbstractQueryGraph


def get_ins_class(t, sid, tgt_objs, mode):
    """
    Get the predicted class for a instance
    @param t:               int, aqg decoding step
    @param sid:             int, sample id
    @param tgt_objs:        (bs, max_tgt_len), ground truth of objects at each aqg decoding step
    @param mode:            "vertex" or "edge"
    @return:                int
    """
    assert mode in ["vertex", "edge"]
    ins_class = tgt_objs[sid][t]
    if mode == "edge":
        # Unified processing of forward and reverse relations
        ins_class = ins_class - 1 if ins_class % 2 == 1 else ins_class
    return ins_class

def get_last_ins_class(t, sid, tgt_objs, mode):
    """
    Get the predicted class for a instance at previous step of the same operation
    @param t:               int, aqg decoding step
    @param sid:             int, sample(question) id
    @param tgt_objs:        (bs, max_tgt_len), ground truth of objects at each aqg decoding step
    @param mode:            "vertex" or "edge"
    @return:                int
    """
    assert mode in ["vertex", "edge"]
    # Operator Sequence: (0)av, (1)av, (2)sv, (3)ae, (4)av, (5)sv, (6)ae, ...
    if mode == "vertex":
        if t == 1:
            last_class = tgt_objs[sid][t - 1]
        else:
            last_class = tgt_objs[sid][t - 3]
    else:
        last_class = tgt_objs[sid][t - 3]
        # Unified processing of both forward and reverse relations
        last_class = last_class - 1 if last_class % 2 == 1 else last_class
    return last_class

def allow_filling_at_t(t, mode):
    """
    Check is filling allowed at step t of AQG decoding
    @param t:           int, aqg decoding step
    @param mode:        "vertex" or "edge"
    @return:            bool
    """
    assert mode in ["vertex", "edge"]
    op = get_operator_by_t(t)
    if mode == "vertex" and op != "av":     # In vertex mode, only handle "AddVertex"
        return False
    if mode == "edge" and op != "ae":       # In edge mode, only handle "AddVertex"
        return False
    return True

def allow_filling_at_t_for_sid(sid, t_ins, tgt_ins_objs):
    """
    Check is filling allowed at step t of AQG decoding
    @param sid:             int, sample(question) id
    @param t_ins:           int, vertex/edge decoding step
    @param tgt_ins_objs:    List(bs), ground truth for vertex/edge instance linking
    @return:                bool
    """
    if t_ins < len(tgt_ins_objs[sid]) and tgt_ins_objs[sid][t_ins] != -1:
        return True
    return False

def allow_copy_at_t_for_sid(t, sid, tgt_objs, tgt_lens, mode):
    """
    Check is copy allowed at step t of AQG decoding
    @param t:               int, aqg decoding step
    @param sid:             int, sample(question) id
    @param tgt_objs:        (bs, max_tgt_len), ground truth of objects at each aqg decoding step
    @param tgt_lens:        (bs)
    @param mode:            "vertex" and "edge"
    @return:                bool
    """
    assert mode in ["vertex", "edge"]
    if mode == "vertex":
        # For vertex, only entity class can be copied
        # print("+++", t, tgt_lens[sid], tgt_objs[sid][t], V_CLASS_IDS["ent"])
        return t < tgt_lens[sid] and tgt_objs[sid][t] == V_CLASS_IDS["ent"]
    else:
        # For edge, only relation class can be copied
        return t < tgt_lens[sid] and tgt_objs[sid][t] in [E_CLASS_IDS["rel+"], E_CLASS_IDS["rel-"]]

def mk_aqg_decoder_input(t, tgt_objs, v_class_embedding, e_class_embedding, zero_embed, v_enc_t):
    """
    Make the input embeddings to the AQG decoder at all steps.
    @param t:                   int, aqg decoding step
    @param tgt_objs:            (bs, max_tgt_len), ground truth of objects at each aqg decoding step
    @param v_class_embedding:   vertex class embedding layer
    @param e_class_embedding:   edge class embedding layer
    @param zero_embed:          (d_h)
    @param v_enc_t:             (bs, n_v, d_h)
    @return:
    """
    bs = tgt_objs.size(0)
    # Make the tgt input embeddings for the current decoding step
    # act_last_embed:           (bs, d_h)
    if t == 0:
        # The embedding of the first token to the decoder
        act_last_embed = zero_embed.unsqueeze(0).expand(bs, -1)
    else:
        last_t = t - 1
        last_op = get_operator_by_t(last_t)
        if last_op == "av":
            # Add Vertex
            act_last_embed = v_class_embedding(tgt_objs[:, last_t])
        elif last_op == "ae":
            # Add Edge
            act_last_embed = e_class_embedding(tgt_objs[:, last_t])
        else:
            # Select Vertex
            act_last_embed = []
            for sid, v_id in enumerate(tgt_objs[:, last_t]):
                act_last_embed.append(v_enc_t[sid][v_id])
            act_last_embed = torch.stack(act_last_embed, dim=0)
    return act_last_embed

def mk_instance_decoder_input(t, tgt_objs, tgt_ins_objs, ins_pool, mode, args):
    """
    Make the input embeddings to the vertex/edge instance decoder at all steps.
    @param t:               int, aqg decoding step
    @param tgt_objs:        (bs, max_tgt_len), ground truth of objects at each aqg decoding step
    @param tgt_ins_objs:    List(bs), ground truth for vertex/edge instance linking
    @param ins_pool:        List(bs), candidate pool of vertex/edge, each element is a DICT {class_id: (total_n_class, d_h)}
    @param mode:            "vertex" and "edge"
    @return:
    """
    assert mode in ["vertex", "edge"]

    bs = len(tgt_objs)
    t_ins = step_to_op_step(t, mode)

    # act_last_embed:       (bs, d_h)
    if t_ins == 0:
        zero_embed = Variable(torch.FloatTensor(args.d_h).zero_(), requires_grad=False)
        if args.cuda:
            zero_embed = zero_embed.to(args.gpu)
        # The embedding of the first token to the decoder
        act_last_embed = zero_embed.unsqueeze(0).expand(bs, -1)
    else:
        # Build the encoding of the last action
        act_last_embed = []
        for sid in range(bs):
            if allow_filling_at_t_for_sid(sid, t_ins - 1, tgt_ins_objs):
                # last vertex instance encoding
                # ins_pool[sid][last_class][last_ins_obj]:  (d_h)
                last_class = get_last_ins_class(t, sid, tgt_objs, mode)
                last_ins_obj = tgt_ins_objs[sid][t_ins - 1]
                act_last_embed.append(ins_pool[sid][last_class][last_ins_obj])
            else:
                zero_embed = Variable(torch.FloatTensor(args.d_h).zero_())
                if args.cuda:
                    zero_embed = zero_embed.to(args.gpu)
                # self.zero_embed:  (d_h)
                act_last_embed.append(zero_embed)
        act_last_embed = torch.stack(act_last_embed, dim=0)
    return act_last_embed

def enhance_decoder_output(h_t, ctx, q_vec, decoder_output_aff, context_mode):
    if context_mode == "attention":
        dec_out = decoder_output_aff(torch.cat([h_t, ctx], -1))
    else:
        dec_out = decoder_output_aff(torch.cat([h_t, q_vec], -1))
    return dec_out

def mk_instance_decoder_output(dec_out, decoder_output_aff, q_vec):
    """
    Combine the output of decoder with question context information.
    @param dec_out:                 (max_tgt_ins_len, bs, d_h)
    @param decoder_output_aff:      decoder_output_aff layer
    @param q_vec:                   (bs, d_h)   question vector
    @param q_vec_for_ins:           (bs, d_h)   question vector by masking mention
    @param args:                    config
    @return:
            dec_out:                (max_tgt_ins_len, bs, d_h)
    """
    bs = dec_out.size(1)
    max_tgt_ins_len = dec_out.size(0)
    # q_vec_exp:        (max_tgt_ins_len, bs, d_h)
    # dec_out:          (max_tgt_ins_len, bs, d_h)
    q_vec_exp = q_vec.unsqueeze(0).expand(max_tgt_ins_len, bs, -1)
    dec_out = decoder_output_aff(torch.cat([dec_out, q_vec_exp], -1))
    return dec_out

def mk_instance_decoder_output2(dec_out, decoder_output_aff, q_vec, q_vec_for_ins, args):
    """
    Combine the output of decoder with question context information.
    @param dec_out:                 (max_tgt_ins_len, bs, d_h)
    @param decoder_output_aff:      decoder_output_aff layer
    @param q_vec:                   (bs, d_h)   question vector
    @param q_vec_for_ins:           (bs, d_h)   question vector by masking mention
    @param args:                    config
    @return:
            dec_out:                (max_tgt_ins_len, bs, d_h)
    """
    bs = dec_out.size(1)
    max_tgt_ins_len = dec_out.size(0)
    if args.use_mention_feature:
        # q_vec_exp:        (max_tgt_ins_len, bs, d_h)
        # dec_out:          (max_tgt_ins_len, bs, d_h)
        q_vec_exp = q_vec.unsqueeze(0).expand(max_tgt_ins_len, bs, -1)
        dec_out = decoder_output_aff(torch.cat([dec_out, q_vec_exp], -1))
    else:
        # q_vec_for_ins_exp:    (max_tgt_ins_len, bs, d_h)
        # dec_out:              (max_tgt_ins_len, bs, d_h)
        q_vec_for_ins_exp = q_vec_for_ins.unsqueeze(0).expand(max_tgt_ins_len, bs, -1)
        dec_out = decoder_output_aff(torch.cat([dec_out, q_vec_for_ins_exp], -1))
    return dec_out

def mk_graph_auxiliary_encoding2(tgt_embeds, cur_g_enc):
    """
    Add the graph information when predicting instances.
    @param tgt_embeds:          (max_tgt_ins_len, bs, d_h), base tgt input embeddings
    @param cur_g_enc:           (bs, d_h)
    @return:                    (max_tgt_ins_len, bs, d_h)
    """
    bs = tgt_embeds.size(1)
    max_tgt_ins_len = tgt_embeds.size(0)
    return cur_g_enc[-1].unsqueeze(0).expand(max_tgt_ins_len, bs, -1)

def combine_graph_auxiliary_encoding(one_tgt_embed, g_enc):
    """
    Add the graph information when predicting instances.
    @param tgt_embeds:          (max_tgt_ins_len, bs, d_h), base tgt input embeddings
    @param cur_g_enc:           (bs, d_h)
    @return:                    (max_tgt_ins_len, bs, d_h)
    """
    one_tgt_embed = torch.add(one_tgt_embed, g_enc)
    return one_tgt_embed

def combine_instance_auxiliary_encoding(one_tgt_embed, ins_enc, t, tgt_aqgs, mode):
    bs = one_tgt_embed.size(0)
    t_ins = step_to_op_step(t, mode)
    aux_ins_enc_t = []
    for sid in range(bs):
        final_aqg = tgt_aqgs[sid][-1]
        if mode == "vertex" and t_ins < final_aqg.vertex_number:
            # Find the vertex id that is added at step t_ins.
            ins_t = final_aqg.get_v_add_history(t_ins)
            aux_ins_enc_t.append(ins_enc[sid][ins_t])  # (d_h)
        elif mode == "edge" and t_ins < final_aqg.edge_number // 2:
            # Find the edge id that is added at step t_ins.
            ins_t = final_aqg.get_e_add_history(t_ins)
            aux_ins_enc_t.append(ins_enc[sid][ins_t])  # (d_h)
        else:
            # Padding with zero embedding
            zero_embed = Variable(torch.FloatTensor(one_tgt_embed.size(-1)).zero_())
            if one_tgt_embed.cuda:
                zero_embed = zero_embed.to(one_tgt_embed.device)
            aux_ins_enc_t.append(zero_embed)  # (d_h)
    aux_ins_enc_t = torch.stack(aux_ins_enc_t, dim=0)
    one_tgt_embed = torch.add(one_tgt_embed, aux_ins_enc_t)
    return one_tgt_embed

def mk_instance_auxiliary_encoding2(tgt_aqgs, cur_ins_enc, zero_embed, mode, t_constraint=10000):
    """
    Add the vertex/edge instance information when predicting instances
    @param tgt_aqgs:            List(bs), original AQG data structure at each step
    @param cur_ins_enc:         (bs, n_ins, d_h)
    @param zero_embed:          (d_h)
    @param mode:                "vertex" and "edge"
    @return:                    (max_tgt_ins_len, bs, d_h)
    """
    assert mode in ["vertex", "edge"]

    bs = len(tgt_aqgs)
    max_tgt_len = max([len(x) for x in tgt_aqgs])

    final_ins_enc = []
    for t in range(max_tgt_len):
        # Check is it filling allowed
        if not allow_filling_at_t(t, mode):
            continue

        t_ins = step_to_op_step(t, mode)

        if t_ins > t_constraint:
            break

        final_ins_enc_t = []
        for sid in range(bs):
            final_aqg = tgt_aqgs[sid][-1]
            if mode == "vertex" and t_ins < final_aqg.vertex_number:
                # Find the vertex id that is added at step t_ins.
                ins_t = final_aqg.get_v_add_history(t_ins)
                final_ins_enc_t.append(cur_ins_enc[sid][ins_t])     # (d_h)
            elif mode == "edge" and t_ins < final_aqg.edge_number // 2:
                # Find the edge id that is added at step t_ins.
                ins_t = final_aqg.get_e_add_history(t_ins)
                final_ins_enc_t.append(cur_ins_enc[sid][ins_t])     # (d_h)
            else:
                # Padding with zero embedding
                final_ins_enc_t.append(zero_embed)                  # (d_h)
        final_ins_enc.append(torch.stack(final_ins_enc_t, dim=0))
    # final_ins_enc:    (max_tgt_ins_len, bs, d_h)
    final_ins_enc = torch.stack(final_ins_enc, dim=0)
    return final_ins_enc

def get_outlining_action_probability(t, dec_out, av_readout, ae_readout, sv_pointer_net, v_enc, args, data):
    """
    Calculate the action_probability of AQG decoding
    @param t:                   int, aqg decoding step
    @param dec_out:             (max_tgt_ins_len, bs, d_h)
    @param av_readout:          linear layer: 2*d_h --> d_h
    @param ae_readout:          linear layer: 2*d_h --> d_h
    @param sv_pointer_net:      pointer network layer
    @param v_enc:               List(max_tgt_len), each element size (bs, n_v, d_h)
    @param args:                config
    @param data:                original data
    @return:
    """
    op = get_operator_by_t(t)
    if op == "av":
        # Add Vertex
        # action_prob:      (bs, V_CLASS_NUM)
        action_prob = av_readout(dec_out[t])

        if args.mask_aqg_prob:
            action_prob = mask_av_action_prob(args.dataset, t, action_prob, data)

    elif op == "ae":
        # Add Edge
        # action_prob:      (bs, E_CLASS_NUM)
        action_prob = ae_readout(dec_out[t])
        if args.mask_aqg_prob:
            action_prob = mask_ae_action_prob(args.dataset, t, action_prob, data)
    else:
        # Select Vertex
        # Cannot select the newly added point
        # action_prob:      (bs, n_v + 1)
        action_prob = sv_pointer_net(src_encodings=v_enc[t],
                                     src_token_mask=mk_sv_mask(v_enc[t]) == 0,
                                     query_vec=dec_out[t].unsqueeze(0))
    return action_prob

def get_segment_switch_action_probability(t, dec_out, seg_readout):
    return seg_readout(dec_out[t])

def get_copy_action_probability(t, dec_out, enc, not_copy_enc, copy_pointer_net):
    bs = dec_out[t].size(0)
    d_h = dec_out[t].size(1)
    # Combine the embeddings and "do not copy"
    # not_copy_enc:     (bs, 1, d_h)
    # copy_enc:         (bs, n + 1, d_h)
    not_copy_enc = not_copy_enc.unsqueeze(0).unsqueeze(0).expand(bs, 1, d_h)
    copy_enc = torch.cat([enc[t], not_copy_enc], dim=1)
    # action_prob:             (bs, n + 1)
    action_prob = copy_pointer_net(src_encodings=copy_enc,
                                   query_vec=dec_out[t].unsqueeze(0),
                                   src_token_mask=None)
    return action_prob

def get_filling_action_probability(t, dec_out, link_pointer_net, tgt_objs, tgt_ins_objs, ins_pool, mode):
    """
    Calculate the action_probability of vertex/edge instance decoding
    @param t:                       int, aqg decoding step
    @param dec_out:                 (max_tgt_ins_len, bs, d_h)
    @param link_pointer_net:        pointer network layer
    @param tgt_objs:                (bs, max_tgt_len), ground truth of objects at each aqg decoding step
    @param tgt_ins_objs:            List(bs), ground truth for vertex/edge instance linking
    @param ins_pool:                List(bs), candidate pool of vertex/edge, each element is a DICT {class_id: (total_n_class, d_h)}
    @param zero_embed:              (d_h)
    @param mode:                    "vertex" or "edge"
    @return:                        (bs, max_n_ins)
    """
    assert mode in ["vertex", "edge"]
    bs = len(tgt_objs)
    t_ins = step_to_op_step(t, mode)

    zero_embed = Variable(torch.FloatTensor(dec_out[0].size(-1)).zero_())
    if dec_out[0].cuda:
        zero_embed = zero_embed.to(dec_out[0].device)

    # Candidate vertex instance encodings
    ins_enc = []
    for sid in range(bs):
        if allow_filling_at_t_for_sid(sid, t_ins, tgt_ins_objs):
            ins_class = get_ins_class(t, sid, tgt_objs, mode)
            ins_enc.append(ins_pool[sid][ins_class])
        else:
            ins_enc.append(zero_embed.unsqueeze(0))

    # ins_enc:        (bs, max_n_ins, d_h)
    # ins_lens:       (bs)
    # ins_mask:       (bs, max_n_ins)      0: True, 1: False
    ins_enc, ins_lens = pad_tensor_1d(ins_enc, 0)
    ins_mask = length_array_to_mask_tensor(ins_lens)

    if dec_out[0].cuda:
        ins_mask = ins_mask.to(dec_out[0].device)

    # action_prob:    (bs, max_n_ins)
    action_prob = link_pointer_net(src_encodings=ins_enc,
                                   src_token_mask=ins_mask,
                                   query_vec=dec_out[t_ins].unsqueeze(0))

    return action_prob

def get_filling_action_probability_for_beams(t, dec_out, beams, link_pointer_net, tgt_objs, tgt_copy_objs,
                                             ins_pool, data, mode, kb, args):

    assert mode in ["vertex", "edge"]
    bs = len(tgt_objs)
    t_ins = step_to_op_step(t, mode)

    zero_embed = Variable(torch.FloatTensor(dec_out[0].size(-1)).zero_())
    if dec_out[0].cuda:
        zero_embed = zero_embed.to(dec_out[0].device)

    # Candidate vertex instance encodings
    ins_enc = []
    for sid in range(bs):
        if t < len(tgt_objs[sid]):
            ins_class = get_ins_class(t, sid, tgt_objs, mode)
            # Predicting
            if (mode == "vertex" and ins_class in ins_pool[sid]) or mode == "edge":
                ins_enc.append(ins_pool[sid][ins_class])
            else:
                ins_enc.append(zero_embed.unsqueeze(0))
        else:
            ins_enc.append(zero_embed.unsqueeze(0))

    # ins_enc:        (bs, max_n_ins, d_h)
    # ins_lens:       (bs)
    # ins_mask:       (bs, max_n_ins)      0: True, 1: False
    ins_enc, ins_lens = pad_tensor_1d(ins_enc, 0)
    ins_mask = length_array_to_mask_tensor(ins_lens)

    if mode == "vertex":
        ins_mask = apply_constraint_for_vertex(t=t,
                                               v_mask=ins_mask,
                                               tgt_objs=tgt_objs,
                                               tgt_copy_objs=tgt_copy_objs,
                                               beams=beams,
                                               ins_pool=ins_pool,
                                               args=args)
    else:
        ins_mask = apply_constraint_for_edge(t=t,
                                             e_mask=ins_mask,
                                             tgt_objs=tgt_objs,
                                             tgt_copy_objs=tgt_copy_objs,
                                             beams=beams,
                                             ins_pool=ins_pool,
                                             data=data,
                                             kb=kb,
                                             args=args)
    if dec_out[0].cuda:
        ins_mask = ins_mask.to(dec_out[0].device)

    # action_prob:    (bs, max_n_ins)
    action_prob = link_pointer_net(src_encodings=ins_enc,
                                   src_token_mask=ins_mask,
                                   query_vec=dec_out[t_ins].unsqueeze(0))

    return action_prob, ins_mask

def apply_constraint_for_vertex(t, v_mask, tgt_objs, tgt_copy_objs, beams, ins_pool, args):
    t_av = step_to_op_step(t, "vertex")

    for bid, beam in enumerate(beams):
        aqg = beam.cur_aqg

        if t_av >= len(tgt_copy_objs[bid]):
            continue

        v_class = get_ins_class(t, bid, tgt_objs, "vertex")
        v_copy = tgt_copy_objs[bid][t_av]

        if v_class in [V_CLASS_IDS["var"], V_CLASS_IDS["ans"], V_CLASS_IDS["end"]]:
            continue

        if args.use_v_copy:
            if v_class == V_CLASS_IDS["ent"] and v_copy != -1:
                for obj in range(len(ins_pool[bid][v_class])):
                    # if use copy mechanism, mask all other instance except for the predicted vertex
                    try:
                        if obj != aqg.get_vertex_instance(v_copy)[0]:
                            v_mask[bid][obj] = 1
                    except:
                        print(t)
                        print(t_av)
                        print(v_class)
                        print(v_copy)
                        print(tgt_objs)
                        print(tgt_copy_objs)
                        aqg.show_state()
                        exit()
            else:
                # Mask vertex instances that have been used
                for obj in range(len(ins_pool[bid][v_class])):
                    for _t_av, used_obj in enumerate(beam.pred_v_ins_objs):
                        _t = av_step_to_step(_t_av)
                        used_v_class = beam.pred_aqg_objs[_t]
                        if used_v_class == v_class and used_obj == obj:
                            v_mask[bid][obj] = 1
                            break
    return v_mask

def apply_constraint_for_edge(t, e_mask, tgt_objs, tgt_copy_objs, beams, ins_pool, data, kb, args):
    t_ae = step_to_op_step(t, "edge")

    for bid, beam in enumerate(beams):
        aqg = beam.cur_aqg

        if t_ae >= len(tgt_copy_objs[bid]):
            continue

        e_class = get_ins_class(t, bid, tgt_objs, "edge")
        e_copy = tgt_copy_objs[bid][t_ae]

        if args.use_e_copy:
            if e_copy != -1:
                for obj in range(len(ins_pool[bid][e_class])):
                    # if use copy mechanism, mask all other instance except for the predicted edge
                    if obj != aqg.get_edge_instance(e_copy)[0]:
                        e_mask[bid][obj] = 1
            else:
                # mask edge instances that have been used
                for obj in range(len(ins_pool[bid][e_class])):
                    for _t_ae, used_obj in enumerate(beam.pred_e_ins_objs):
                        _t = ae_step_to_step(_t_ae)
                        used_e_class = beam.pred_aqg_objs[_t] - 1 if beam.pred_aqg_objs[_t] % 2 == 1 else beam.pred_aqg_objs[_t]
                        if used_e_class == e_class and used_obj == obj:
                            e_mask[bid][obj] = 1
                            break

        if e_class in [E_CLASS_IDS["rel+"], E_CLASS_IDS["rel-"]]:
            e_mask = mk_constraint_for_relation(t_ae=t_ae,
                                                bid=bid,
                                                e_mask=e_mask,
                                                e_class=e_class,
                                                aqg=aqg,
                                                dataset=args.dataset,
                                                data=data[bid])

            if args.use_kb_constraint:
                e_mask = execution_guided_constraint(t_ae=t_ae,
                                                     bid=bid,
                                                     e_mask=e_mask,
                                                     e_class=e_class,
                                                     aqg=aqg,
                                                     data=data[bid],
                                                     kb=kb,
                                                     kb_endpoint=args.kb_endpoint,
                                                     use_subgraph=args.use_subgraph)

        if e_class in [E_CLASS_IDS["agg+"], E_CLASS_IDS["agg-"]]:
            e_mask = mk_constraint_for_aggregation(bid=bid,
                                                   e_mask=e_mask,
                                                   e_class=e_class,
                                                   dataset=args.dataset,
                                                   data=data[bid])

    return e_mask

def mk_constraint_for_relation(t_ae, bid, e_mask, e_class, aqg, dataset, data):
    eid = aqg.get_e_add_history(t_ae)
    # make constraint for <var, rdf:type, type>
    if dataset == "lcq":
        now_s, now_o = -1, -1
        for j, (s, o, p) in enumerate(aqg.triples):
            if p == eid:
                now_s, now_o = s, o
                break
        assert now_s != -1 and now_o != -1
        now_s_class = aqg.get_vertex_label(now_s)
        now_o_class = aqg.get_vertex_label(now_o)
        if now_s_class == V_CLASS_IDS["type"] or now_o_class == V_CLASS_IDS["type"]:
            for i, (e_ins_name, _) in enumerate(data["instance_pool"]["edge"][e_class]):
                if e_ins_name == "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>":
                    e_mask[bid][i] = 0
                else:
                    e_mask[bid][i] = 1
        else:
            for i, (e_ins_name, _) in enumerate(data["instance_pool"]["edge"][e_class]):
                if e_ins_name == "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>":
                    e_mask[bid][i] = 1
                else:
                    e_mask[bid][i] = 0

    # make constraint for <var1, rel, var2> <var2, cmp, date>
    if dataset in ["wsp", "cwq"]:
        date_vars = set()
        for j, triple in enumerate(aqg.triples):
            s, o, p = triple
            _p_class = aqg.get_edge_label(p)
            if _p_class not in [E_CLASS_IDS["cmp+"], E_CLASS_IDS["cmp-"]]:
                continue
            _s_class = aqg.get_vertex_label(s)
            _o_class = aqg.get_vertex_label(o)
            if _s_class in [V_CLASS_IDS["var"], V_CLASS_IDS["ans"]] and _o_class in [V_CLASS_IDS["var"], V_CLASS_IDS["ans"]]:
                date_vars.add(s)
                date_vars.add(o)
            else:
                if _s_class in [V_CLASS_IDS["var"], V_CLASS_IDS["ans"]]:
                    _o_name = aqg.get_vertex_instance(o)[-1]
                    if "xsd:dateTime" in _o_name:
                        date_vars.add(s)
                if _o_class in [V_CLASS_IDS["var"], V_CLASS_IDS["ans"]]:
                    _s_name = aqg.get_vertex_instance(s)[-1]
                    if "xsd:dateTime" in _s_name:
                        date_vars.add(o)

        now_s, now_o = -1, -1
        for j, triple in enumerate(aqg.triples):
            s, o, p = triple
            if p == eid:
                now_s, now_o = s, o
                break

        # print("---", e_mask[bid])

        assert now_s != -1 and now_o != -1
        if now_s in date_vars or now_o in date_vars:
            for i, (e_ins_name, _) in enumerate(data["instance_pool"]["edge"][e_class]):
                if e_ins_name.split(".")[-1] not in ["from", "to", "from$$$to", "start_date",
                                                 "end_date", "start_date$$$end_date"]:
                    e_mask[bid][i] = 1
        else:
            for i, (e_ins_name, _) in enumerate(data["instance_pool"]["edge"][e_class]):
                if e_ins_name.split(".")[-1] in ["from", "to", "from$$$to", "start_date",
                                                 "end_date", "start_date$$$end_date"]:
                    e_mask[bid][i] = 1
    return e_mask

def execution_guided_constraint(t_ae, bid, e_mask, e_class, aqg, data, kb, kb_endpoint, use_subgraph):
    n_timeout = 0
    eid = aqg.get_e_add_history(t_ae)
    # print("++++++++++++", t_ae, bid)
    # aqg.show_state()
    for i, (e_ins_name, _) in enumerate(data["instance_pool"]["edge"][e_class]):
        if e_mask[bid][i] == 1:
            continue
        tmp_aqg = copy.deepcopy(aqg)
        tmp_aqg.set_edge_instance(eid, [i, e_ins_name])
        tmp_aqg.set_edge_instance(get_inv_edge(eid), [i, e_ins_name])
        try:
            if not use_subgraph:
                tmp_queries = tmp_aqg.to_temporary_sparql_query(kb=kb)
            else:
                tmp_queries = tmp_aqg.to_ask_sparql_query_for_eg(kb=kb, qid=str(data["id"]))

            if not tmp_queries:
                e_mask[bid][i] = 1
                continue

            for one_query in tmp_queries:
                # print(one_query)
                result = KB_query_with_timeout(one_query, kb_endpoint)
                # print(result)
                # print()

                if result == "TimeOut":
                    result = [False]
                    n_timeout += 1
                # print(src_mask[b_id])
                if not result[0]:
                    e_mask[bid][i] = 1
                    break
            if n_timeout >= 3:
                for j in range(i, len(data["instance_pool"]["edge"][e_class])):
                    e_mask[bid][j] = 1
                break
        except:
            # if self.args.dataset != "lcq":
            #     src_mask[b_id][i] = 1
            e_mask[bid][i] = 1
            continue
    return e_mask

def mk_constraint_for_aggregation(bid, e_mask, e_class, dataset, data):
    if dataset == "cwq":
        for i, (e_ins_name, _) in enumerate(data["instance_pool"]["edge"][e_class]):
            if e_ins_name in ["ASK"]:
                e_mask[bid][i] = 1
    if dataset == "lcq":
        for i, (e_ins_name, _) in enumerate(data["instance_pool"]["edge"][e_class]):
            if e_ins_name in ["MAX", "MIN"]:
                e_mask[bid][i] = 1
    return e_mask

def mask_av_action_prob(dataset, t, action_prob, data):
    """
    Make the mask for AddVertex operation.
    @param dataset:         dataset name
    @param t:               int, aqg decoding step
    @param action_prob:     (bs, V_CLASS_NUM)
    @param data:            original data
    @return:                (bs, V_CLASS_NUM)
    """
    bs = action_prob.size(0)
    assert action_prob.size(1) == len(V_CLASS_IDS)
    if t == 0:
        # At first step, only allow to choose "ans" class
        # mask:     (bs, V_CLASS_NUM)
        mask = np.ones((bs, len(V_CLASS_IDS)), dtype=np.uint8)
        for i in range(bs):
            mask[i][V_CLASS_IDS["ans"]] = 0
        mask = torch.ByteTensor(mask)
    else:
        # mask:     (bs, V_CLASS_NUM)
        mask = np.zeros((bs, len(V_CLASS_IDS)), dtype=np.uint8)
        for i in range(bs):
            mask[i][V_CLASS_IDS["ans"]] = 1
            for o_id in [V_CLASS_IDS["type"], V_CLASS_IDS["ent"]]:
                # Mask the class that does not have any candidate instance.
                if o_id not in data[i]["instance_pool"]["vertex"]:
                    mask[i][o_id] = 1
            if dataset == "lcq":
                # LC-QuAD does not have "val" class.
                mask[i][V_CLASS_IDS["val"]] = 1
        mask = torch.ByteTensor(mask)
    if action_prob.is_cuda:
        mask = mask.to(action_prob.device)
    action_prob.masked_fill_(mask.bool(), -float('inf'))
    return action_prob

def mask_ae_action_prob(dataset, t, action_prob, data):
    """
    Make the mask for AddEdge operation.
    @param dataset:         dataset name
    @param t:               int, aqg decoding step
    @param action_prob:     (bs, E_CLASS_NUM)
    @param data:            original data
    @return:                (bs, E_CLASS_NUM)
    """
    bs = action_prob.size(0)
    assert action_prob.size(1) == len(E_CLASS_IDS)
    # mask:       (bs, E_CLASS_NUM)
    mask = np.zeros((bs, len(E_CLASS_IDS)), dtype=np.uint8)
    for i in range(bs):
        for o_id in [E_CLASS_IDS["agg+"], E_CLASS_IDS["cmp+"], E_CLASS_IDS["ord+"], E_CLASS_IDS["rel+"]]:
            # Mask the class that does not have any candidate instance.
            if o_id not in data[i]["instance_pool"]["edge"]:
                mask[i][o_id] = 1
                mask[i][o_id + 1] = 1

        if dataset == "lcq":
            # LC-QuAD does not have "cmp" class and "ord" class.
            mask[i][E_CLASS_IDS["cmp+"]] = 1
            mask[i][E_CLASS_IDS["cmp+"] + 1] = 1
            mask[i][E_CLASS_IDS["ord+"]] = 1
            mask[i][E_CLASS_IDS["ord+"] + 1] = 1

    mask = torch.ByteTensor(mask)
    if action_prob.is_cuda:
        mask = mask.to(action_prob.device)
    action_prob.masked_fill_(mask.bool(), -float('inf'))
    return action_prob

def mk_sv_mask(one_v_enc):
    """
    Make the mask for SelectVertex operation, 0: Mask, 1: Not mask
    @param one_v_enc:       (bs, n_v, d_h)
    @return:                (bs, n_v + 1, d_h)
    """
    # sv_mask:              (bs, n_v + 1, d_h)
    # The newly added vertex at previous step can not be chosen at current step
    sv_mask = torch.cat([torch.LongTensor(one_v_enc.size(0), one_v_enc.size(1) - 1).fill_(1),
                         torch.LongTensor(one_v_enc.size(0), 1).zero_()], -1)
    if one_v_enc.cuda:
        sv_mask = sv_mask.to(one_v_enc.device)
    return sv_mask

def get_av_obj_range(dataset, one_data):
    """
    During generation, provide the choices for AddVertex operation
    @param dataset:     dataset name
    @param one_data:    one original data
    @return:            List
    """
    obj_range = []
    for obj in range(len(V_CLASS_IDS)):
        # Except for the first vertex, there will be no "ans" vertex.
        if obj == V_CLASS_IDS["ans"]:
            continue
        # The "obj" class must have one instance at least.
        if obj not in [V_CLASS_IDS["end"], V_CLASS_IDS["var"]] and obj not in one_data["instance_pool"]["vertex"]:
            continue
        # LC-QuAD dataset does not have "val" vertices.
        if obj == V_CLASS_IDS["val"] and dataset == "lcq":
            continue
        obj_range.append(obj)
    return obj_range

def get_ae_obj_range(dataset, one_data):
    """
    During generation, provide the choices for AddEdge operation
    @param dataset:     dataset name
    @param one_data:    one original data
    @return:            List
    """
    obj_range = []
    for obj in range(len(E_CLASS_IDS)):
        # If "+" direction class does not have instances, skip
        if obj % 2 == 0 and obj not in one_data["instance_pool"]["edge"] and obj + 1 not in one_data["instance_pool"]["edge"]:
            continue
        # If "-" direction class does not have instances, skip
        if obj % 2 == 1 and obj not in one_data["instance_pool"]["edge"] and obj - 1 not in one_data["instance_pool"]["edge"]:
            continue
        # LC-QuAD dataset does not have "cmp" and "ord" edges.
        if dataset == "lcq" and obj in [E_CLASS_IDS["cmp+"],
                                        E_CLASS_IDS["cmp-"],
                                        E_CLASS_IDS["ord+"],
                                        E_CLASS_IDS["ord-"]]:
            continue
        obj_range.append(obj)
    return obj_range

def get_sv_obj_range(aqg):
    """
    During generation, provide the choices for SelectVertex operation
    @param aqg:   AbstractQueryGraph
    @return:      List
    """
    return [obj for obj in range(aqg.vertex_number - 1)]

def get_segment_switch_range(dataset, aqg: AbstractQueryGraph, n_v_start):
    """
    During generation, provide the choices for Switch Segment mechanism.
    @param dataset:         dataset name
    @param aqg:             AbstractQueryGraph
    @param n_v_start:       The number of vertices that starts to check whether switch segment.
    @return:                List
    """
    # Enumerate when to switch segment
    if dataset == "cwq" and aqg.vertex_number >= n_v_start:
        seg_switch_range = [0, 1]
    else:
        seg_switch_range = [0]
    return seg_switch_range

def get_vertex_copy_range(aqg: AbstractQueryGraph, v_class, use_v_copy):
    """
    During generation, provide the choices for Copy mechanism.
    @param aqg:             AbstractQueryGraph
    @param v_class:         class of vertex v
    @param use_v_copy:      bool, whether use the Copy mechanism
    @return:
    """
    n_v = aqg.vertex_number
    # Enumerate the copied vertex, only when the class is entity
    if use_v_copy and v_class == V_CLASS_IDS["ent"]:
        copy_range = [v for v in range(n_v + 1)]
    else:
        # Only retain the last embedding, denoting "do not copy vertex"
        copy_range = [n_v]
    new_copy_range = []
    for v in copy_range:
        if v != n_v and aqg.get_vertex_label(v) != V_CLASS_IDS["ent"]:
            continue
        new_copy_range.append(v)
    return new_copy_range

def get_edge_copy_range(aqg: AbstractQueryGraph, e_class, use_e_copy):
    """
    During generation, provide the choices for Copy mechanism.
    @param aqg:             AbstractQueryGraph
    @param e_class:         class of edge e
    @param use_e_copy:      bool, whether use the Copy mechanism
    @return:                List
    """
    n_e = aqg.edge_number
    # Enumerate the copy edge
    if use_e_copy and e_class in [E_CLASS_IDS["rel+"], E_CLASS_IDS["rel-"]]:
        #: TODO: check aqg.edge_number
        copy_range = [e for e in range(n_e) if e < aqg.edge_number] + [n_e]
    else:
        # Only the last encoding, denoting "do not copy edge"
        copy_range = [n_e]
    return copy_range

def mk_meta_entries(t, dec_out, beams, n_beams, av_readout, ae_readout, sv_pointer_net, seg_readout,
                        not_copy_enc, copy_pointer_net,  v_enc, e_enc, args, data):
    """
    Collect all the possible expansions of the current beams by the action probabilities from the decoder
    @param t:                       int, aqg decoding step
    @param dec_out:                 (max_tgt_ins_len, bs, d_h)
    @param beams:                   List(beam_sz)
    @param av_readout:              linear layer
    @param ae_readout:              linear layer
    @param sv_pointer_net:          Pointer network layer
    @param seg_readout:             linear layer
    @param not_copy_enc:            embeddings denoting that "do not copy"
    @param copy_pointer_net:        Pointer network layer
    @param v_enc:                   List(t + 1), each element size (beam_sz, n_v, d_h)
    @param e_enc:                   List(t + 1), each element size (bs, n_v, d_h)
    @param args:                    args
    @param data:                    List(1), original data
    @return:    meta_entries:       the possible expansions of the current beams
    """
    op = get_operator_by_t(t)

    # data_exp:         List(beam_sz)

    # action_prob:      (beam_sz, max_act_len)
    action_prob = get_outlining_action_probability(t=t,
                                                   dec_out=dec_out,
                                                   av_readout=av_readout,
                                                   ae_readout=ae_readout,
                                                   sv_pointer_net=sv_pointer_net,
                                                   v_enc=v_enc,
                                                   args=args,
                                                   data=data)

    action_prob = F.log_softmax(action_prob, dim=-1)

    if op == "av":

        # copy_action_prob:             (beam_sz, max_n_v)
        # seg_switch_action_prob:       (beam_sz, 2)
        copy_action_prob = get_copy_action_probability(t=t,
                                                       dec_out=dec_out,
                                                       enc=v_enc,
                                                       not_copy_enc=not_copy_enc("vertex"),
                                                       copy_pointer_net=copy_pointer_net("vertex"))
        seg_switch_action_prob = get_segment_switch_action_probability(t=t,
                                                                       dec_out=dec_out,
                                                                       seg_readout=seg_readout)

        copy_action_prob = F.log_softmax(copy_action_prob, dim=-1)
        seg_switch_action_prob = F.log_softmax(seg_switch_action_prob, dim=-1)

        meta_entries = mk_meta_entries_for_av(t=t,
                                              beams=beams,
                                              n_beams=n_beams,
                                              action_prob=action_prob,
                                              copy_action_prob=copy_action_prob,
                                              seg_switch_action_prob=seg_switch_action_prob,
                                              data=data,
                                              args=args)

    elif op == "ae":
        # copy_action_prob:             (beam_sz, max_n_e)
        copy_action_prob = get_copy_action_probability(t=t,
                                                       dec_out=dec_out,
                                                       enc=e_enc,
                                                       not_copy_enc=not_copy_enc("edge"),
                                                       copy_pointer_net=copy_pointer_net("edge"))

        copy_action_prob = F.log_softmax(copy_action_prob, dim=-1)

        meta_entries = mk_meta_entries_for_ae(t=t,
                                              beams=beams,
                                              n_beams=n_beams,
                                              action_prob=action_prob,
                                              copy_action_prob=copy_action_prob,
                                              data=data,
                                              args=args)
    else:
        meta_entries = mk_meta_entries_for_sv(t=t,
                                              beams=beams,
                                              n_beams=n_beams,
                                              action_prob=action_prob)

    return meta_entries

def mk_meta_entries_for_av(t, beams, n_beams, action_prob, copy_action_prob, seg_switch_action_prob, data, args):
    """
    Collect all the possible expansions of the current beams by performing AddVertex operation.
    @param t:                           int, aqg decoding step
    @param beams:                       List(beam_sz)
    @param action_prob:                 (beam_sz, max_act_len)
    @param copy_action_prob:            (beam_sz, max_n_v)
    @param seg_switch_action_prob:      (beam_sz, 2)
    @param one_data:                    one original data
    @param args:                        config
    @return:                            List
    """
    bs = len(n_beams)
    meta_entries = [[] for _ in range(bs)]                   # Each element is a recording of the updating of the AQG.
    op = get_operator_by_t(t)

    for bid, beam in enumerate(beams):
        aqg = beam.cur_aqg
        sid = beam.sid

        if t == 0:
            # First vertex is always in the "ans" class
            meta_entry = {
                "op": op,
                "obj": V_CLASS_IDS["ans"],
                "obj_score": action_prob[bid, V_CLASS_IDS["ans"]],
                "seg": 0,
                "seg_score": None,
                "v_cp": -1,
                "v_cp_score": None,
                "e_cp": -1,
                "e_cp_score": None,
                "aqg_score": aqg.get_score() + action_prob[bid, V_CLASS_IDS["ans"]].detach().cpu().numpy(),
                "prev_beam_id": bid
            }
            meta_entries[sid].append(meta_entry)
        else:
            # Get the possible choices of AddVertex
            obj_range = get_av_obj_range(args.dataset, data[bid])
            for obj in obj_range:
                # Update score
                new_aqg_score = aqg.get_score() + action_prob[bid, obj].cpu().detach().numpy()

                # Finish the AQG generation
                if obj == V_CLASS_IDS["end"]:
                    meta_entry = {
                        "op": op,
                        "obj": obj,
                        "obj_score": action_prob[bid, obj],
                        "seg": 0,
                        "seg_score": None,
                        "v_cp": -1,
                        "v_cp_score": None,
                        "e_cp": -1,
                        "e_cp_score": None,
                        "aqg_score": new_aqg_score,
                        "prev_beam_id": bid
                    }
                    meta_entries[sid].append(meta_entry)
                    continue

                # Get the possible choices of Segment Switch.
                seg_switch_range = get_segment_switch_range(args.dataset, aqg, args.v_num_start_switch_segment)
                for seg in seg_switch_range:
                    new_aqg_score_1 = new_aqg_score
                    if args.dataset == "cwq":
                        # Update score by adding segment switching probability
                        new_aqg_score_1 += seg_switch_action_prob[bid, seg].cpu().detach().numpy()

                    # Get the possible choices of Vertex Copy.
                    copy_range = get_vertex_copy_range(aqg, obj, args.use_v_copy)
                    for v in copy_range:
                        new_aqg_score_2 = new_aqg_score_1
                        # Update score by adding vertex copy probability
                        if args.use_v_copy:
                            new_aqg_score_2 += copy_action_prob[bid, v].cpu().detach().numpy()

                        meta_entry = {
                            "op": op,
                            "obj": obj,
                            "obj_score": action_prob[bid, obj],
                            "seg": seg,
                            "seg_score": seg_switch_action_prob[bid, seg],
                            "v_cp": v if v != aqg.vertex_number else -1,
                            "v_cp_score": copy_action_prob[bid, v],
                            "e_cp": -1,
                            "e_cp_score": None,
                            "aqg_score": new_aqg_score_2,
                            "prev_beam_id": bid
                        }
                        meta_entries[sid].append(meta_entry)

    return meta_entries

def mk_meta_entries_for_ae(t, beams, n_beams, action_prob, copy_action_prob, data, args):
    """
    Collect all the possible expansions of the current beams by performing AddEdge operation.
    @param t:                       int, aqg decoding step
    @param beams:                   List(beam_sz)
    @param action_prob:             (beam_sz, max_act_len)
    @param copy_action_prob:        (beam_sz, max_n_e)
    @param one_data:                one original data
    @param args:                    config
    @return:                        List
    """
    bs = len(n_beams)
    meta_entries = [[] for _ in range(bs)]
    op = get_operator_by_t(t)

    for bid, beam in enumerate(beams):
        aqg = beam.cur_aqg
        sid = beam.sid

        # Get the possible choices of AddEdge
        obj_range = get_ae_obj_range(args.dataset, data[bid])
        for obj in obj_range:
            # Update action score
            new_aqg_score = aqg.get_score() + action_prob[bid, obj].cpu().detach().numpy()

            # Get the possible choices of Edge Copy
            copy_range = get_edge_copy_range(aqg, obj, args.use_e_copy)

            for e in copy_range:
                new_aqg_score_1 = new_aqg_score
                # Update score by adding edge copying probability
                if args.use_e_copy:
                    new_aqg_score_1 += copy_action_prob[bid, e].cpu().detach().numpy()

                meta_entry = {
                    "op": op,
                    "obj": obj,
                    "obj_score": action_prob[bid, obj],
                    "seg": 0,
                    "seg_score": None,
                    "v_cp": -1,
                    "v_cp_score": None,
                    "e_cp": e if e != aqg.edge_number else -1,
                    "e_cp_score": copy_action_prob[bid, e],
                    "aqg_score": new_aqg_score_1,
                    "prev_beam_id": bid
                }
                meta_entries[sid].append(meta_entry)
    return meta_entries

def mk_meta_entries_for_sv(t, beams, n_beams, action_prob):
    """
    Collect all the possible expansions of the current beams by performing SelectVertex operation.
    @param t:                   int, aqg decoding step
    @param beams:               List(beam_sz)
    @param action_prob:         (beam_sz, max_act_len)
    @return:                    List
    """
    bs = len(n_beams)
    meta_entries = [[] for _ in range(bs)]
    op = get_operator_by_t(t)

    for bid, beam in enumerate(beams):
        aqg = beam.cur_aqg
        sid = beam.sid

        # Enumerate vertex in current AQG to select
        obj_range = get_sv_obj_range(aqg)

        for obj in obj_range:
            # Update action score
            new_aqg_score = aqg.get_score() + action_prob[bid, obj].cpu().detach().numpy()
            meta_entry = {
                "op": op,
                "obj": obj,
                "obj_score": action_prob[bid, obj],
                "seg": 0,
                "seg_score": None,
                "v_cp": -1,
                "v_cp_score": None,
                "e_cp": -1,
                "e_cp_score": None,
                "aqg_score": new_aqg_score,
                "prev_beam_id": bid
            }
            meta_entries[sid].append(meta_entry)
    return meta_entries

def mk_filling_meta_entries(t, dec_out, beams, n_beams, completed_beams, max_beam_sz, link_pointer_net, tgt_objs, tgt_copy_objs, ins_pool, data, args, kb, mode):
    bs = len(n_beams)
    t_ins = step_to_op_step(t, mode)


    # action_prob:    (bs, max_n_ins)
    action_prob, ins_mask = get_filling_action_probability_for_beams(t=t,
                                                                     dec_out=dec_out,
                                                                     beams=beams,
                                                                     link_pointer_net=link_pointer_net,
                                                                     tgt_objs=tgt_objs,
                                                                     tgt_copy_objs=tgt_copy_objs,
                                                                     ins_pool=ins_pool,
                                                                     data=data,
                                                                     mode=mode,
                                                                     kb=kb,
                                                                     args=args)


    action_prob = F.log_softmax(action_prob, dim=-1)

    tmp_beams = []
    tmp_live_beam_ids = []
    tmp_n_beams = [0 for _ in range(bs)]

    meta_entries = [[] for _ in range(bs)]

    for bid, beam in enumerate(beams):
        aqg = beam.cur_aqg

        if t >= len(tgt_objs[bid]):
            continue

        if mode == "vertex":
            sid = beam.sid_v
            ins_class = tgt_objs[bid][t]

            if ins_class in [V_CLASS_IDS["var"], V_CLASS_IDS["ans"], V_CLASS_IDS["end"]]:
                beam.add_vertex_instance_object(-1)

                if t == len(tgt_objs[bid]) - 1 and len(completed_beams[sid]) < max_beam_sz:
                    completed_beams[sid].append(beam)
                    continue

                tmp_n_beams[sid] += 1
                tmp_beams.append(beam)
                tmp_live_beam_ids.append(bid)
                continue
        else:
            sid = beam.sid_e
            ins_class = tgt_objs[bid][t] - 1 if tgt_objs[bid][t] % 2 == 1 else tgt_objs[bid][t]

        # enumerate the vertex instance
        for obj in range(len(ins_pool[bid][ins_class])):

            if ins_mask[bid][obj] == 1:
                continue

            # update probability
            if mode == "vertex":
                new_ins_score = aqg.get_v_score() + action_prob[bid, obj].cpu().detach().numpy()
                ins_idx = aqg.get_v_add_history(t_ins)
            else:
                new_ins_score = aqg.get_e_score() + action_prob[bid, obj].cpu().detach().numpy()
                ins_idx = aqg.get_e_add_history(t_ins)

            meta_entry = {
                mode: ins_idx,
                "obj": obj,
                "obj_score": action_prob[bid, obj],
                "new_ins_score": new_ins_score,
                "prev_beam_id": bid
            }
            meta_entries[sid].append(meta_entry)

    # assert len(meta_entries) * len(new_beams) == 0
    return meta_entries, tmp_beams, tmp_n_beams, tmp_live_beam_ids

def organize_outlining_beams(t, beams, meta_entries, completed_beams, max_beam_sz, args, data):
    """
    Organize the beams in Outlining, to select the top-k possible expansions as the new beams.
    @param t:                       int, aqg decoding step
    @param beams:                   List(beam_sz), original beams
    @param meta_entries:            List, possible expansions of beams
    @param completed_beams:         List, beams that finish Outlining
    @param max_beam_sz:             maximum size of beam
    @param args:                    config
    @param one_data:                one original data
    @return:      new_beams:        List of new beams
    """

    bs = len(meta_entries)

    new_beams = []
    live_beam_ids = []
    new_n_beams = [0 for _ in range(bs)]

    for sid in range(bs):

        # aqg_scores:       (meta_entries_len)
        aqg_scores = torch.FloatTensor([x["aqg_score"] for x in meta_entries[sid]])

        # Select top-k aqg with highest probs
        # top_aqg_scores:   (k)
        # entry_ids:        (k)
        k = min(aqg_scores.size(0), max_beam_sz - len(completed_beams[sid]))
        top_aqg_scores, entry_ids = torch.topk(aqg_scores, k=k)

        one_n_beams = 0
        one_new_beams = []
        one_live_beam_ids = []

        for score, idx in zip(top_aqg_scores.cpu().detach().numpy(), entry_ids.data.cpu()):
            meta_entry = meta_entries[sid][idx]
            prev_beam = beams[meta_entry["prev_beam_id"]]       # Previous beam

            # Build a new beam with AQG update
            beam = copy.deepcopy(prev_beam)
            beam.update_step(t)
            beam.update_previous_beam_id(meta_entry["prev_beam_id"])

            aqg = copy.deepcopy(beam.cur_aqg)
            aqg.update_score(score)

            # Update beam
            if meta_entry["op"] == "av" and meta_entry["obj"] == V_CLASS_IDS["end"]:
                # Outlining is finished.
                if args.dataset == "lcq":
                    if aqg.check_final_structure(data[sid]["instance_pool"], args.dataset):
                        # Check the final structure of AQG is legal.
                        beam.add_aqg(aqg)
                        beam.add_object(meta_entry["obj"])
                        if t > 1:
                            completed_beams[sid].append(beam) # beam
                        # print("xxx", len(completed_beams[sid]))
                else:
                    if aqg.check_final_structure(data[sid]["instance_pool"], args.dataset):
                        beam.add_aqg(aqg)
                        beam.add_object(meta_entry["obj"])
                        if t > 1:
                            completed_beams[sid].append(beam)
                        # print("yyy", len(completed_beams[sid]))
            else:
                # Outlining is finished, update AQG state
                if meta_entry["op"] == "av":
                    aqg.update_state("av", [meta_entry["obj"], meta_entry["v_cp"], meta_entry["seg"]])
                    beam.add_vertex_copy_object(meta_entry["v_cp"])
                elif meta_entry["op"] == "ae":
                    aqg.update_state("ae", [meta_entry["obj"], meta_entry["e_cp"]])
                    beam.add_edge_copy_object(meta_entry["e_cp"])
                else:
                    aqg.update_state("sv", meta_entry["obj"])

                # Update beam and recode the previous AQGs and predicted objects.
                beam.add_aqg(aqg)
                beam.add_object(meta_entry["obj"])

                if args.dataset == "lcq" and not aqg.check_temporary_structure(args.dataset):
                    continue

                one_n_beams += 1
                one_new_beams.append(beam)
                one_live_beam_ids.append(meta_entry["prev_beam_id"])

        if len(completed_beams[sid]) >= max_beam_sz:
            new_n_beams[sid] = 0
        else:
            new_n_beams[sid] = one_n_beams
            new_beams.extend(one_new_beams)
            live_beam_ids.extend(one_live_beam_ids)

    return new_beams, new_n_beams, live_beam_ids

def organize_filling_beams(t, beams, tgt_objs, meta_entries, completed_beams, max_beam_sz, data, mode):
    bs = len(meta_entries)

    new_beams = []
    new_n_beams = []
    live_beam_ids = []

    for sid in range(bs):
        # print(sid, len(meta_entries[sid]))

        ins_scores = torch.FloatTensor([x["new_ins_score"] for x in meta_entries[sid]])

        # select top-k beams with highest probs
        k = min(ins_scores.size(0), max_beam_sz - len(completed_beams[sid]))

        top_ins_scores, entry_ids = torch.topk(ins_scores, k=k)

        one_n_beams = 0
        one_new_beams = []
        one_live_beam_ids = []

        for _, idx in zip(top_ins_scores.data.cpu(), entry_ids.data.cpu()):
            meta_entry = meta_entries[sid][idx]
            ins = meta_entry[mode]
            obj = meta_entry["obj"]
            prev_bid = meta_entry['prev_beam_id']

            beam = copy.deepcopy(beams[prev_bid])

            if mode == "vertex":
                beam.update_vertex_previous_beam_id(prev_bid)
            else:
                beam.update_edge_previous_beam_id(prev_bid)

            aqg = copy.deepcopy(beam.cur_aqg)

            if mode == "vertex":
                aqg.update_v_score(meta_entry["new_ins_score"])
                ins_class = beam.pred_aqg_objs[t]
            else:
                aqg.update_e_score(meta_entry["new_ins_score"])
                ins_class = beam.pred_aqg_objs[t] - 1 if beam.pred_aqg_objs[t] % 2 == 1 else beam.pred_aqg_objs[t]


            ins_name, _ = data[sid]["instance_pool"][mode][ins_class][obj]

            if mode == "vertex":
                # set the predicted instance to vertex v
                aqg.set_vertex_instance(v=ins, v_instance=[obj, ins_name])
                beam.add_vertex_instance_object(obj)
            else:
                aqg.set_edge_instance(e=ins, e_instance=[obj, ins_name])
                aqg.set_edge_instance(e=get_inv_edge(ins), e_instance=[obj, ins_name])
                beam.add_edge_instance_object(obj)

            beam.add_aqg(aqg)

            one_n_beams += 1
            one_new_beams.append(beam)
            one_live_beam_ids.append(meta_entry["prev_beam_id"])

            if len(completed_beams[sid]) < max_beam_sz:
                if (mode == "vertex" and t == len(tgt_objs[prev_bid]) - 1) or \
                        (mode == "edge" and t == len(tgt_objs[prev_bid]) - 2):
                    completed_beams[sid].append(beam)

        if len(completed_beams[sid]) >= max_beam_sz:
            new_n_beams.append(0)
        else:
            new_n_beams.append(one_n_beams)
            new_beams.extend(one_new_beams)
            live_beam_ids.extend(one_live_beam_ids)


    return new_beams, new_n_beams, live_beam_ids

def mk_tgt_from_beams(beams, args, mode="AQG"):
    """
    During Outlining, make the target objects and target aqgs as the inputs to the decoders
    @param beams:           List(beam_sz), original beams
    @param args:            config
    @return:
    """
    tgt_aqgs = []
    tgt_aqg_objs = []
    tgt_aqg_inputs = []
    tgt_v_ins_objs = []
    tgt_e_ins_objs = []

    tgt_v_copy_objs = []
    tgt_e_copy_objs = []

    for bid, beam in enumerate(beams):
        # print("###", beam.sid)
        # Transform the AQG to the inputs embeddings to the graph encoder
        one_inputs = [mk_graph_for_gnn(*aqg.get_state()) for aqg in beam.pred_aqgs]

        tgt_aqgs.append(copy.deepcopy(beam.pred_aqgs))
        tgt_aqg_inputs.append(one_inputs)
        tgt_aqg_objs.append([x for x in beam.pred_aqg_objs])
        tgt_v_ins_objs.append([x for x in beam.pred_v_ins_objs])
        tgt_e_ins_objs.append([x for x in beam.pred_e_ins_objs])

        tgt_v_copy_objs.append([x for x in beam.pred_v_copy_objs])
        tgt_e_copy_objs.append([x for x in beam.pred_e_copy_objs])

    # tgt_aqg_inputs:       List(t + 1)     AQG embeddings (vertex and edges) input to graph encoder,
    #                                       each element: (v, v_class, v_segment, e, e_class, e_segment, adj)
    # tgt_aqg_lens:         (beam_sz)
    # tgt_aqg_objs:         (beam_sz, t)
    # tgt_aqg_obj_lens:     (beam_sz)
    # For each i, tgt_aqg_obj_lens[i] + 1 = tgt_aqg_lens[i]

    if args.cuda:
        tgt_aqg_inputs = [[[y.to(args.gpu) for y in g] for g in s] for s in tgt_aqg_inputs]

    if mode == "AQG":
        return tgt_aqg_objs, tgt_aqg_inputs
    elif mode == "vertex":
        return tgt_aqgs, tgt_aqg_objs, tgt_aqg_inputs,\
               tgt_v_ins_objs, tgt_v_copy_objs,
    else:
        return tgt_aqgs, tgt_aqg_objs, tgt_aqg_inputs, \
               tgt_e_ins_objs, tgt_e_copy_objs

def check_filling(aqg, mode):
    if mode == "vertex":
        for v in aqg.vertices:
            if aqg.v_labels[v] in [V_CLASS_IDS["ans"], V_CLASS_IDS["var"]]:
                continue
            if v not in aqg.v_instances:
                return False
    else:
        for e in aqg.edges:
            if e not in aqg.e_instances:
                return False
    return True

def init_zero_decoder_state(q_enc):
    bs = q_enc.size(0)
    d_h = q_enc.size(-1)
    h0 = torch.autograd.Variable(torch.FloatTensor(bs, d_h).zero_())
    c0 = torch.autograd.Variable(torch.FloatTensor(bs, d_h).zero_())
    if q_enc.cuda:
        h0 = h0.to(q_enc.device)
        c0 = c0.to(q_enc.device)
    return [h0, c0]

def pad_input_embeds_for_gnn(v_embeds, e_embeds, adjs):

    # print([x.size() for x in e_embeds])
    v_embeds, _ = pad_tensor_1d(v_embeds, pad_idx=0)
    e_embeds, _ = pad_tensor_1d(e_embeds, pad_idx=0)

    max_n_v = v_embeds.size(1)
    max_n_e = e_embeds.size(1)
    max_sz = max_n_v + max_n_e + 1

    new_adjs = []
    padding_mask = []

    for sid, adj in enumerate(adjs):
        sz = adj.size(0)
        one_padding_mask = Variable(torch.FloatTensor(max_sz).zero_())
        if sz < max_sz:
            # print(adj.size())
            zero_embed_1 = Variable(torch.FloatTensor(max_sz - sz, sz).zero_())
            zero_embed_2 = Variable(torch.FloatTensor(max_sz, max_sz - sz).zero_())
            if v_embeds.cuda:
                zero_embed_1 = zero_embed_1.to(v_embeds.device)
                zero_embed_2 = zero_embed_2.to(v_embeds.device)
            adj = torch.cat([adj, zero_embed_1], dim=0)
            adj = torch.cat([adj, zero_embed_2], dim=1)
            for i in range(sz, max_sz):
                adj[i][i] = 1
                one_padding_mask[i] = 1
        new_adjs.append(adj)
        padding_mask.append(one_padding_mask)

    new_adjs = torch.stack(new_adjs, dim=0)
    padding_mask = torch.stack(padding_mask, dim=0)
    return v_embeds, e_embeds, new_adjs, padding_mask

def pad_output_embeds_for_gnn(v_enc, e_enc, g_enc, select_sids):
    st = 0
    pad_v_enc = []
    pad_e_enc = []
    pad_g_enc = []
    for i in range(len(select_sids)):
        for j in range(st, select_sids[i]):
            pad_v_enc.append(torch.zeros_like(v_enc[0]))
            pad_e_enc.append(torch.zeros_like(e_enc[0]))
            pad_g_enc.append(torch.zeros_like(g_enc[0]))
        if i != len(select_sids) - 1:
            pad_v_enc.append(v_enc[i])
            pad_e_enc.append(e_enc[i])
            pad_g_enc.append(g_enc[i])
        st = select_sids[i] + 1
    pad_v_enc = torch.stack(pad_v_enc, dim=0)
    pad_e_enc = torch.stack(pad_e_enc, dim=0)
    pad_g_enc = torch.stack(pad_g_enc, dim=0)
    return pad_v_enc, pad_e_enc, pad_g_enc

def expand_tensor_by_beam_number(tensor, n_beams):
    bs = tensor.size(0)
    n_dim = len(tensor.size())
    n_exp_sz = [[n_beams[sid]] + [1 for _ in range(n_dim - 1)] for sid in range(bs)]

    exp_q_enc = []
    for sid in range(bs):
        exp_q_enc.append(tensor[sid].repeat(n_exp_sz[sid]))
    exp_q_enc = torch.cat(exp_q_enc, dim=0)
    return exp_q_enc

def expand_graph_embedding_by_beam_number(g_enc, ins_enc, n_beams):
    bs = len(g_enc)
    exp_g_enc = []
    exp_ins_enc = []
    for sid in range(bs):
        exp_g_enc.append(g_enc[sid].unsqueeze(0).repeat(n_beams[sid], 1))
        exp_ins_enc.append(ins_enc[sid].unsqueeze(0).repeat(n_beams[sid], 1, 1))
    exp_g_enc = torch.cat(exp_g_enc, dim=0)
    exp_ins_enc, _ = pad_tensor_2d(exp_ins_enc, pad_idx=0)
    return exp_g_enc, exp_ins_enc

def expand_data_by_beam_number(datas, n_beams):
    new_datas = []
    bs = len(datas)
    for sid in range(bs):
        new_datas.extend([datas[sid] for _ in range(n_beams[sid])])
    return new_datas

def is_complete(completed_beams, max_beam_sz):
    count = sum([1 if len(x) == max_beam_sz else 0 for x in completed_beams])
    # print("---", count, len(completed_beams))
    if count == len(completed_beams):
        return True
    else:
        return False

def mk_input_by_schedule(context, q_enc, q_vec, q_mask, g_enc, ins_enc, ins_pool, data, beams, n_beams, schedules, is_finished, mode):
    bs = len(beams)
    in_beams = []
    for sid in range(bs):
        if schedules[sid] < n_beams[sid] and not is_finished[sid]:
            # print("---", sid)
            in_beams.append(copy.deepcopy(beams[sid][schedules[sid]]))

    if mode == "vertex":
        for sid_v, beam in enumerate(in_beams):
            beam.sid_v = sid_v
    else:
        for sid_e, beam in enumerate(in_beams):
            beam.sid_e = sid_e

    index = []
    in_ins_pool = []
    in_data = []
    in_g_enc = []
    in_ins_enc = []
    for beam in in_beams:
        index.append(beam.sid)
        in_ins_pool.append(ins_pool[beam.sid])
        in_data.append(data[beam.sid])

        # print("+++", beam.t, beam.prev_beam_id)
        # print(len(g_enc), [x.size() for x in g_enc])
        in_g_enc.append(g_enc[beam.t][beam.prev_beam_id])
        in_ins_enc.append(ins_enc[beam.t][beam.prev_beam_id])

    index = torch.LongTensor(index)
    if q_enc.cuda:
        index = index.to(q_enc.device)
    in_context = context.index_select(dim=0, index=index)
    in_q_enc = q_enc.index_select(dim=0, index=index)
    in_q_vec = q_vec.index_select(dim=0, index=index)
    in_q_mask = q_mask.index_select(dim=0, index=index)

    return in_context, in_q_enc, in_q_vec, in_q_mask, in_ins_pool, in_data, in_g_enc, in_ins_enc, in_beams

def check_schedule(schedules, is_finished, n_beams):
    for sid in range(len(is_finished)):
        if not is_finished[sid] and schedules[sid] < n_beams[sid]:
            return False
    return True

def update_schedule(schedules, in_beams, out_beams, finished_beams):
    for beam in in_beams:
        schedules[beam.sid] += 1
    for sid_v in range(len(out_beams)):
        for b in out_beams[sid_v]:
            finished_beams[out_beams[sid_v][0].sid].append(b)

def update_schedule_with_kb_constraint(schedules, in_beams, out_beams, finished_beams, dataset, kb, kb_endpoint):
    for beam in in_beams:
        schedules[beam.sid] += 1
    for sid_v in range(len(out_beams)):
        for b in out_beams[sid_v]:
            b.cur_aqg.normalize(dataset)
            query = b.cur_aqg.to_final_sparql_query(kb=kb)
            try:
                answers = KB_query(query, kb_endpoint)
                if 'ASK' not in query and 'COUNT' not in query:  # ask
                    answers = [r["x_0"] for r in answers]
            except:
                answers = []
            if len(answers) > 0:
                finished_beams[out_beams[sid_v][0].sid].append(b)

def initial_finish_state(finished_beams, is_finished):
    for sid in range(len(is_finished)):
        if not is_finished[sid]:
            finished_beams[sid] = []

def update_finish_state(finished_beams, is_finished, final_beams):
    for sid in range(len(finished_beams)):
        if finished_beams[sid] and not is_finished[sid]:
            is_finished[sid] = True
            final_beams[sid].extend(finished_beams[sid])

def combine_and_sort_by_sid(tmp_beams, beams, tmp_live_beam_ids, live_beam_ids, tmp_n_beams, n_beams, mode):
    bs = len(n_beams)
    assert len(tmp_n_beams) == bs

    n_beams = [n_beams[i] + tmp_n_beams[i] for i in range(bs)]

    beams = tmp_beams + beams
    live_beam_ids = tmp_live_beam_ids + live_beam_ids
    assert len(beams) == len(live_beam_ids)

    tmp = [[beams[i], live_beam_ids[i]] for i in range(len(beams))]
    # for x, y in tmp:
    #     print(x.sid_v)
    #     print(y)
    # print("------")

    if mode == "vertex":
        tmp.sort(key=lambda x: x[0].sid_v)
    else:
        tmp.sort(key=lambda x: x[0].sid_e)
    # for x, y in tmp:
    #     print(x.sid_v)
    #     print(y)
    # print("+++++")

    beams = [x[0] for x in tmp]
    live_beam_ids = [x[1] for x in tmp]

    # for i, x in enumerate(beams):
    #     print(x.sid_v, live_beam_ids[i])
    # print("----------------")

    return beams, n_beams, live_beam_ids

def get_final_graph_embeddings(g_embeds, ins_embeds, tgt_objs):
    index = [len(x) - 1 for x in tgt_objs]
    bs = len(g_embeds[0])
    final_g_embeds = []
    final_ins_embeds = []
    for i in range(bs):
        final_g_embeds.append(g_embeds[index[i]][i])
        final_ins_embeds.append(ins_embeds[index[i]][i])
    final_g_embeds = torch.stack(final_g_embeds, dim=0)

    # print(final_g_embeds.size())
    # print([x.size() for x in final_ins_embeds])
    final_ins_embeds, _ = pad_tensor_1d(final_ins_embeds, pad_idx=0)
    # print(final_ins_embeds.size())
    # exit()
    return final_g_embeds, final_ins_embeds