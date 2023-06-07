import os
import json
import argparse
import re


def align_step(data):
    alignments = []
    for data_idx, d in enumerate(data):
        recipe_steps = []
        for step in d['recipe']['procedure']:
            tmp = step['summary'] + ' ' + step['description']
            tmp = re.sub(' +', ' ', tmp)
            tmp = re.sub(r'\n', ' ', tmp)
            recipe_steps.append(tmp.strip())
        current_step_idx = 0
        turn_alignments = []
        for turn_idx, turn in enumerate(d['conversation']):
            if turn['id'] == 'Agent':
                evidences = turn['evidence']
                if len(evidences) == 0:
                    turn_alignments.append(current_step_idx)
                else:
                    aligned_step = -1
                    for evidence in evidences:
                        for idx, step in enumerate(recipe_steps):
                            if evidence in step:
                                aligned_step = idx
                                break
                    if aligned_step == -1:
                        # print(data_idx, turn_idx, evidences)
                        aligned_step = current_step_idx
                    turn_alignments.append(aligned_step)
                    current_step_idx = aligned_step
        alignments.append(turn_alignments)
    return alignments


def main(args):
    all_alignments = {}
    splits = ['train', 'val', 'test']
    data_path = args.input_dir
    for split in splits:
        f_names = []
        all_data = []
        all_alignments[split] = {}
        for f_name in os.listdir(os.path.join(data_path, split)):
            absolute_f_path = os.path.join(data_path, split, f_name)
            with open(absolute_f_path) as f:
                data = json.load(f)
            data_id = f'{split}_{f_name}'
            data['id'] = data_id
            if len(data['conversation']) > 2:
                all_data.append(data)
                f_names.append(f_name)
            else:
                print('skip: ', f_name)
        alignments = align_step(all_data)
        for f_name, alignment in zip(f_names, alignments):
            all_alignments[split][f_name] = alignment
    with open(args.output_file, 'w') as f:
        json.dump(all_alignments, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()
    main(args)
