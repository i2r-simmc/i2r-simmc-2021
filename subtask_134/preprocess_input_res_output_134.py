#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.
This source code is licensed under the license found in the LICENSE file in the
root directory of this source tree.

    Scripts for converting the main SIMMC datasets (.JSON format)
    into the line-by-line stringified format (and back).

    The reformatted data is used as input for the GPT-2 based
    DST model baseline.

There are some mismatches between actual data and data descriptions.
"act" corresponds to "intent"
"slot-values" and "request_slots" correspond to slots
"objects" corresponds to  scene object ids
"""
# from gpt2_dst.utils.convert import convert_json_to_flattened
import argparse
import json
import os
from copy import deepcopy

# DSTC style dataset fieldnames
FIELDNAME_DIALOG = "dialogue"
FIELDNAME_USER_UTTR = "transcript"
FIELDNAME_ASST_UTTR = "system_transcript"
FIELDNAME_BELIEF_STATE = "transcript_annotated"
FIELDNAME_SYSTEM_STATE = "system_transcript_annotated"

# Templates for GPT-2 formatting
START_OF_MULTIMODAL_CONTEXTS = "<SOM>"
END_OF_MULTIMODAL_CONTEXTS = "<EOM>"
START_BELIEF_STATE = "=>"
START_OF_RESPONSE = "<SOR>"
END_OF_BELIEF = "<EOB>"
END_OF_SENTENCE = "<EOS>"

TEMPLATE_PREDICT = "{context} {START_BELIEF_STATE}"
TEMPLATE_TARGET = (
    "{belief_state} {END_OF_BELIEF} {response} {END_OF_SENTENCE}"
)

# No belief state predictions and target.
TEMPLATE_PREDICT_NOBELIEF = "{context} {START_BELIEF_STATE}"
TEMPLATE_TARGET_NOBELIEF = "{response} {END_OF_SENTENCE}"


def represent_visual_objects(object_ids):
    str_objects = ", ".join([str(o) for o in object_ids])
    return f"{START_OF_MULTIMODAL_CONTEXTS} {str_objects} {END_OF_MULTIMODAL_CONTEXTS}"


def convert_json_to_flattened(
    input_path_json,
    input_meta_json_folder,
    input_scene_path_json,
    output_path_predict,
    output_path_target,
    len_context=2,
    use_multimodal_contexts=False,
    use_belief_states=False,
    use_meta_contexts=False,
    input_path_special_tokens="",
    output_path_special_tokens="",
):
    """
    Input: JSON representation of the dialogs
    Output: line-by-line stringified representation of each turn
    """

    with open(input_path_json, "r") as f_in:
        data = json.load(f_in)["dialogue_data"]
    input_meta_data_path_json = {
        'fashion': input_meta_json_folder+'/fashion_prefab_metadata_all.json',
        'furniture': input_meta_json_folder+'/furniture_prefab_metadata_all.json'
        }
    meta_data = {}
    for meta_domain in input_meta_data_path_json:
        meta_data_path = input_meta_data_path_json[meta_domain]
        with open(meta_data_path, "r") as f_in:
            meta_data[meta_domain] = json.load(f_in)
    predicts = []
    targets = []
    if input_path_special_tokens != "":
        with open(input_path_special_tokens, "r") as f_in:
            special_tokens = json.load(f_in)
    else:
        special_tokens = {"eos_token": END_OF_SENTENCE}
        additional_special_tokens = []
        if use_belief_states:
            additional_special_tokens.append(END_OF_BELIEF)
        else:
            additional_special_tokens.append(START_OF_RESPONSE)
        additional_special_tokens.extend(
                [START_OF_MULTIMODAL_CONTEXTS, END_OF_MULTIMODAL_CONTEXTS]
            )
        special_tokens["additional_special_tokens"] = additional_special_tokens

    if output_path_special_tokens != "":
        # If a new output path for special tokens is given,
        # we track new OOVs
        oov = set()

    for _, dialog in enumerate(data):

        prev_asst_uttr = None
        prev_turn = None
        lst_context = []
        temp_scene_ids = [int(x) for x in dialog["scene_ids"]]
        temp_scene_ids.sort()
        temp_scene_ids.append(len(dialog["dialogue"]))
        scene_ids = []
        domain=dialog["domain"]
        for first, second in zip(temp_scene_ids, temp_scene_ids[1:]):
            assert str(first) in dialog["scene_ids"]
            for i in range(first, second):
                scene_ids.append(dialog["scene_ids"][str(first)])
        scenes_json_path = [os.path.join(input_scene_path_json, scene_id + "_scene.json")
                            for scene_id in scene_ids]
        for turn_id, turn in enumerate(dialog[FIELDNAME_DIALOG]):
            user_uttr = turn[FIELDNAME_USER_UTTR].replace("\n", " ").strip()
            user_belief = turn[FIELDNAME_BELIEF_STATE]
            asst_uttr = turn[FIELDNAME_ASST_UTTR].replace("\n", " ").strip()
            disambiguation_label = turn.get("disambiguation_label", '')
            # Format main input context
            context = ""
            if prev_asst_uttr:
                context += f"System : {prev_asst_uttr} "
                if use_multimodal_contexts:
                    # Add multimodal contexts
                    visual_objects = prev_turn[FIELDNAME_SYSTEM_STATE][
                        "act_attributes"
                    ]["objects"]
                    context += represent_visual_objects(visual_objects) + " "

            context += f"User : {user_uttr}"
            prev_asst_uttr = asst_uttr
            prev_turn = turn

            # Add multimodal contexts -- user shouldn't have access to ground-truth
            """
            if use_multimodal_contexts:
                visual_objects = turn[FIELDNAME_BELIEF_STATE]['act_attributes']['objects']
                context += ' ' + represent_visual_objects(visual_objects)
            """

            # Concat with previous contexts
            lst_context.append(context)
            context = " ".join(lst_context[-len_context:])
            # print(use_belief_states)
            # print(use_meta_contexts)
            # input()
            # Format belief state
            if use_belief_states and use_meta_contexts:
                belief_state = []
                # for bs_per_frame in user_belief:
                str_belief_state_per_frame = (
                    "{act} [ {slot_values} ] ({request_slots}) < {disambige_label} >".format(
                        act=user_belief["act"].strip(),
                        slot_values=", ".join(
                            [
                                f"{k.strip()} = {str(v).strip()}"
                                for k, v in user_belief["act_attributes"][
                                "slot_values"
                            ].items()
                            ]
                        ),
                        request_slots=", ".join(
                            user_belief["act_attributes"]["request_slots"]
                        ),
                        disambige_label=disambiguation_label
                    )
                )
                belief_state.append(str_belief_state_per_frame)

                # Track OOVs
                if output_path_special_tokens != "":
                    oov.add(user_belief["act"])
                    for slot_name in user_belief["act_attributes"]["slot_values"]:
                        oov.add(str(slot_name))
                        # slot_name, slot_value = kv[0].strip(), kv[1].strip()
                        # oov.add(slot_name)
                        # oov.add(slot_value)

                str_belief_state = " ".join(belief_state)
                with open(scenes_json_path[turn_id], "r") as f_in:
                    scenes_json_data=json.load(f_in)
                object_info = scenes_json_data['scenes'][0]['objects']
                final_object_info = deepcopy(object_info)
                for i in range(len(object_info)):
                    final_object_info[i]['meta_data'] = meta_data[domain][final_object_info[i]['prefab_path']]
                all_object_info_list=[]
                for each_object_info in final_object_info:
                    meta_data_object=each_object_info["meta_data"]
                    # object_str=" ".join([meta_data_object['type'],
                    #                      meta_data_object['color'],
                    #                      meta_data_object['brand']])
                    object_str=" ".join([meta_data_object['type']])
                    all_object_info_list.append(object_str)

                str_all_object_info = " ".join(all_object_info_list)
                predict = TEMPLATE_PREDICT.format(
                    context=context,
                    START_BELIEF_STATE=START_OF_MULTIMODAL_CONTEXTS +
                                       " "+str_all_object_info + " " +
                                       END_OF_MULTIMODAL_CONTEXTS
                )
                predicts.append(predict)

                # Format the main output
                target = TEMPLATE_TARGET.format(
                    context=context,
                    START_BELIEF_STATE=START_BELIEF_STATE,
                    belief_state=str_belief_state,
                    END_OF_BELIEF=END_OF_BELIEF,
                    response=asst_uttr,
                    END_OF_SENTENCE=END_OF_SENTENCE,
                )
                targets.append(target)



            else:
                # Format the main input
                predict = TEMPLATE_PREDICT_NOBELIEF.format(
                    context=context, START_OF_RESPONSE=START_OF_RESPONSE
                )
                predicts.append(predict)

                # Format the main output
                target = TEMPLATE_TARGET_NOBELIEF.format(
                    context=context,
                    response=asst_uttr,
                    END_OF_SENTENCE=END_OF_SENTENCE,
                    START_OF_RESPONSE=START_OF_RESPONSE,
                )
                targets.append(target)

    # Create a directory if it does not exist
    directory = os.path.dirname(output_path_predict)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    directory = os.path.dirname(output_path_target)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    # Output into text files
    with open(output_path_predict, "w") as f_predict:
        X = "\n".join(predicts)
        f_predict.write(X)

    with open(output_path_target, "w") as f_target:
        Y = "\n".join(targets)
        f_target.write(Y)

    if output_path_special_tokens != "":
        # Create a directory if it does not exist
        directory = os.path.dirname(output_path_special_tokens)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        with open(output_path_special_tokens, "w") as f_special_tokens:
            # Add oov's (acts and slot names, etc.) to special tokens as well
            special_tokens["additional_special_tokens"].extend(list(oov - set(special_tokens['additional_special_tokens'])))
            json.dump(special_tokens, f_special_tokens)


if __name__ == "__main__":
    # Parse input args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path_json", help="input path to the original dialog data"
    )
    parser.add_argument(
        "--input_meta_json_folder", help="input path to the original dialog data"
    )
    parser.add_argument(
        "--input_scene_path_json", help="input path to the original dialog data"
    )
    parser.add_argument("--output_path_predict", help="output path for model input")
    parser.add_argument("--output_path_target", help="output path for full target")
    parser.add_argument(
        "--input_path_special_tokens",
        help="input path for special tokens. blank if not provided",
        default="",
    )
    parser.add_argument(
        "--output_path_special_tokens",
        help="output path for special tokens. blank if not saving",
        default="",
    )
    parser.add_argument(
        "--len_context",
        help="# of turns to include as dialog context",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--use_multimodal_contexts",
        help="determine whether to use the multimodal contexts each turn",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--no_belief_states",
        dest="use_belief_states",
        action="store_false",
        default=True,
        help="determine whether to use belief state for each turn",
    )
    parser.add_argument(
        "--use_meta_contexts",
        help="determine whether to use the multimodal contexts each turn",
        type=int,
        default=1,
    )

    args = parser.parse_args()
    input_path_json = args.input_path_json
    output_path_predict = args.output_path_predict
    output_path_target = args.output_path_target
    input_meta_json_folder= args.input_meta_json_folder
    input_scene_path_json= args.input_scene_path_json

    input_path_special_tokens = args.input_path_special_tokens
    output_path_special_tokens = args.output_path_special_tokens
    len_context = args.len_context
    use_multimodal_contexts = bool(args.use_multimodal_contexts)
    use_meta_contexts = bool(args.use_meta_contexts)

    # DEBUG:
    print("Belief states: {}".format(args.use_belief_states))

    # Convert the data into GPT-2 friendly format
    convert_json_to_flattened(
        input_path_json,
        input_meta_json_folder,
        input_scene_path_json,
        output_path_predict,
        output_path_target,
        input_path_special_tokens=input_path_special_tokens,
        output_path_special_tokens=output_path_special_tokens,
        len_context=len_context,
        use_multimodal_contexts=use_multimodal_contexts,
        use_belief_states=args.use_belief_states,
        use_meta_contexts=use_meta_contexts
    )
