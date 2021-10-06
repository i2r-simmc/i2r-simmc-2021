import os
import sys
import json
from collections import defaultdict
import argparse

# DSTC style dataset fieldnames
FIELDNAME_DIALOG = "dialogue"
FIELDNAME_USER_UTTR = "transcript"
FIELDNAME_ASST_UTTR = "system_transcript"
FIELDNAME_BELIEF_STATE = "transcript_annotated"
FIELDNAME_SYSTEM_STATE = "system_transcript_annotated"
OBJECT_ATTR = ["prefab_path", "unique_id", "index", "bbox", "position"]
FASHION_METADATA_NON_VISUAL_ATTR = ["customerReview", "brand", "price", "size"]
FURNITURE_METADATA_NON_VISUAL_ATTR = ["customerRating", "brand", "price", "materials"]

# Templates for GPT-2 formatting
START_OF_MULTIMODAL_CONTEXTS = "<SOM>"
END_OF_MULTIMODAL_CONTEXTS = "<EOM>"

def represent_visual_objects(object_ids):
    # Stringify visual objects (JSON)
    str_objects = ", ".join([str(o) for o in object_ids])
    
    return f"Objects: {str_objects}"

def read_meta_data(meta_file):
    with open(meta_file, "r") as inf:
        objects_metadata = json.load(inf)

    return objects_metadata

def read_scenes(scene_folder):
    scenes = {}
    files = os.listdir(scene_folder)
    for filename in files:
        if "scene" in filename:
            with open(os.path.join(scene_folder, filename), "r") as inf:
                scene = json.load(inf)["scenes"]
            scene = scene[0]
            scene_name = filename.split('.')[0][:-6]
            scenes[scene_name] = scene

    return scenes

def combine_scene_metadata(scenes, fashion_metadata, furniture_metadata):
    scene_metadata = {}
    scene_objects_index = {}
    for scene_name, scene in scenes.items():
        objects = scene["objects"]
        object_metas = {}
        objects_index = []
        for obj in objects:
            object_id = obj["prefab_path"]
            unique_id = obj["unique_id"]
            index = obj["index"]
            bbox = obj["bbox"]
            position = obj["position"]
            objects_index.append(index)
            if object_id in fashion_metadata:
                customerReview = fashion_metadata[object_id]["customerReview"]
                brand = fashion_metadata[object_id]["brand"]
                price = fashion_metadata[object_id]["price"]
                size = fashion_metadata[object_id]["size"]
                obj_meta = {"prefab_path": object_id,
                            "unique_id": unique_id,
                            "index": index,
                            "bbox": bbox,
                            "position": position,
                            "customerReview": customerReview,
                            "brand": brand,
                            "price": price,
                            "size": size,
                            "domain": "fashion"
                            }
                object_metas[index] = obj_meta
            elif object_id in furniture_metadata:
                customerRating = furniture_metadata[object_id]["customerRating"]
                brand = furniture_metadata[object_id]["brand"]
                price = furniture_metadata[object_id]["price"]
                materials = furniture_metadata[object_id]["materials"]
                obj_meta = {"prefab_path": object_id,
                            "unique_id": unique_id,
                            "index": index,
                            "bbox": bbox,
                            "position": position,
                            "customerRating": customerRating,
                            "brand": brand,
                            "price": price,
                            "materials": materials,
                            "domain": "furniture"
                            }
                object_metas[index] = obj_meta
            else:
                print("!ERROR! Can't find {} in meta data!".format(object_id))
        scene_metadata[scene_name] = object_metas
        scene_objects_index[scene_name] = objects_index

    return scene_metadata, scene_objects_index


if __name__ == '__main__':
    # Parse input args
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--input_path_json", help="input path to the original dialog data, including train, dev, and devtest json files"
            )
    parser.add_argument(
            "--scene_folder", default="./simmc2_data/public", help="scene folder for current input")
    parser.add_argument(
            "--output_path", help="output path for model input")
    parser.add_argument(
            "--eval_target_path", help="file path for evaluation following SIMMC2")
    parser.add_argument(
            "--len_context",
            help="# of turns to include as dialog context",
            type=int,
            default=1
            )
    parser.add_argument(
            "--use_object_relationships",
            help="whether use object relationships in scenes",
            type=bool,
            default=False
            )
    parser.add_argument(
            "--mode",
            help="train, dev, or test mode",
            type=str,
            default="test"
            )
    args = parser.parse_args()
    input_path_json = args.input_path_json
    output_path = args.output_path
    eval_target_path = args.eval_target_path
    len_context = args.len_context
    use_object_relationships = args.use_object_relationships
    mode = args.mode
    
    fashion_meta_file = './simmc2_data/fashion_prefab_metadata_all.json'
    furniture_meta_file = './simmc2_data/furniture_prefab_metadata_all.json'
    scene_folder = args.scene_folder

    fashion_metadata = read_meta_data(fashion_meta_file)
    furniture_metadata = read_meta_data(furniture_meta_file)

    scenes = read_scenes(scene_folder)

    scene_metadata, scene_objects_index = combine_scene_metadata(scenes, fashion_metadata, furniture_metadata)
    
    with open(input_path_json, "r") as inf:
        data = json.load(inf)["dialogue_data"]

    candidate_object_nums = []
    total_candidate_object_num = 0
    groundtruth_object_nums = []
    scene_nums = []

    model_input_data = []
    eval_target_data = []

    for dialog_i, dialog in enumerate(data):
        prev_asst_uttr = None
        prev_turn = None
        lst_context = []

        dialog_idx = dialog["dialogue_idx"]
        scene_ids = dialog["scene_ids"]
        scene_ids_sorted = sorted(scene_ids.items(), key=lambda x: int(x[0]))
        scene_ids_len = len(scene_ids_sorted)
        scene_nums.append(scene_ids_len)

        for turn in dialog[FIELDNAME_DIALOG]:
            turn_idx = turn["turn_idx"]
            user_uttr = turn[FIELDNAME_USER_UTTR].replace("\n", " ").strip()
            user_belief = {}
            if FIELDNAME_BELIEF_STATE in turn:
                user_belief = turn[FIELDNAME_BELIEF_STATE]
            asst_uttr = ''
            if FIELDNAME_ASST_UTTR in turn:
                asst_uttr = turn[FIELDNAME_ASST_UTTR].replace("\n", " ").strip()
            asst_system = turn[FIELDNAME_SYSTEM_STATE]
            user_objects = []
            if FIELDNAME_BELIEF_STATE in turn:
                user_objects = user_belief["act_attributes"]["objects"]
            asst_objects = asst_system["act_attributes"]["objects"]

            flag = 0
            for i in range(scene_ids_len):
                start_turn_idx = int(scene_ids_sorted[i][0])
                if turn_idx < start_turn_idx:
                    flag = 1
                    break
            if flag:
                scene_name = scene_ids_sorted[i-1][1]
                current_scene_id = i - 1
            else:
                scene_name = scene_ids_sorted[scene_ids_len-1][1]
                current_scene_id = scene_ids_len - 1

            # Format main input context
            context = ""
            if prev_asst_uttr:
                context += f"System : {prev_asst_uttr} "
                visual_objects = prev_turn[FIELDNAME_SYSTEM_STATE]["act_attributes"]["objects"]
                context += represent_visual_objects(visual_objects) + " "
            
            context += f"User : {user_uttr}"
            prev_asst_uttr = asst_uttr
            prev_turn = turn

            # Concat with previous contexts
            lst_context.append(context)
            context = "\t".join(lst_context[-len_context:])
            
            # Format responses and labels and model input data
            # Find candidate objects
            turn_object_metadata = scene_metadata[scene_name]
            if current_scene_id:
                previous_scene_name = scene_ids_sorted[current_scene_id-1][1]
                previous_turns_object_metadata = scene_metadata[previous_scene_name]
                for index, obj_meta in previous_turns_object_metadata.items():
                    if index not in turn_object_metadata:
                        turn_object_metadata[index] = obj_meta
            candidate_objects_metadata = sorted(turn_object_metadata.items(), key=lambda x: x[0])
            
            if mode != "test":
                for obj_index in user_objects:
                    responses = []
                    labels = []
                    label = 1
                    labels.append(label)
                    response = ""
                    for attr in OBJECT_ATTR:
                        response = response + f"{attr}: {turn_object_metadata[obj_index][attr]} "
                    if turn_object_metadata[obj_index]["domain"] == "fashion":
                        for attr in FASHION_METADATA_NON_VISUAL_ATTR:
                            response = response + f"{attr}: {turn_object_metadata[obj_index][attr]} "
                    if turn_object_metadata[obj_index]["domain"] == "furniture":
                        for attr in FURNITURE_METADATA_NON_VISUAL_ATTR:
                            response = response + f"{attr}: {turn_object_metadata[obj_index][attr]} "
                    response = response.strip()
                    responses.append(response)

                    candidate_object_num = 0    
                    for cand_index, object_meta in candidate_objects_metadata:
                        if cand_index not in user_objects:
                            candidate_object_num += 1
                            label = 0
                            labels.append(label)
                            response = ""
                            for attr in OBJECT_ATTR:
                                response = response + f"{attr}: {object_meta[attr]} "
                            if object_meta["domain"] == "fashion":
                                for attr in FASHION_METADATA_NON_VISUAL_ATTR:
                                    response = response + f"{attr}: {object_meta[attr]} "
                            if object_meta["domain"] == "furniture":
                                for attr in FURNITURE_METADATA_NON_VISUAL_ATTR:
                                    response = response + f"{attr}: {object_meta[attr]} "
                            response = response.strip()
                            responses.append(response)
                    object_num_in_one_case = len(labels)
                    
                    candidate_object_nums.append(candidate_object_num)
                    for object_i in range(object_num_in_one_case):
                        model_input_data.append({"Label": labels[object_i],
                                                 "Context": context,
                                                 "Responses": responses[object_i]
                                                 }
                                                )
                
                groundtruth_object_nums.append(len(user_objects))
            
            else:
                responses = []
                labels = []
                candidate_object_ids = []
                candidate_object_nums.append(len(candidate_objects_metadata))
                total_candidate_object_num += len(candidate_objects_metadata)
                current_cand_num = 0
                for cand_index, object_meta in candidate_objects_metadata:
                    candidate_object_ids.append(cand_index)
                    if current_cand_num == 0:
                        label = 1
                    else:
                        label = 0
                    labels.append(label)
                    response = ""
                    for attr in OBJECT_ATTR:
                        response = response + f"{attr}: {object_meta[attr]} "
                        if object_meta["domain"] == "fashion":
                            for attr in FASHION_METADATA_NON_VISUAL_ATTR:
                                response = response + f"{attr}: {object_meta[attr]} "
                        if object_meta["domain"] == "furniture":
                            for attr in FURNITURE_METADATA_NON_VISUAL_ATTR:
                                response = response + f"{attr}: {object_meta[attr]} "
                    response = response.strip()
                    responses.append(response)
                    current_cand_num += 1
                for object_i in range(len(candidate_objects_metadata)):
                    model_input_data.append({"Label": labels[object_i],
                                             "Context": context,
                                             "Responses": responses[object_i]
                                            }
                                           )
                eval_target_data.append({"dialogue_idx": dialog_idx,
                                         "turn_idx": turn_idx,
                                         "transcript_annotated": {
                                             "act": "",
                                             "act_attributes": {
                                                 "slot_values": {},
                                                 "request_slots": [],
                                                 "objects": user_objects
                                                 }
                                             },
                                         "candidate_objects": sorted(candidate_object_ids)
                                         }
                                        )

    with open(output_path, "w") as outf:
        for input_data in model_input_data:
            outf.write('{}\t{}\t{}\n'.format(str(input_data["Label"]), input_data["Context"], input_data["Responses"]))
    if mode == 'test':
        with open(eval_target_path, "w") as outf:
            for eval_target_exp in eval_target_data:
                json.dump(eval_target_exp, outf)
                outf.write("\n")
