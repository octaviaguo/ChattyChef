import os
import json
import argparse

def main(args):
    all_predictions = {}
    for f_name in os.listdir(args.input_dir):
        with open(os.path.join(args.input_dir, f_name)) as f:
            predictions = json.load(f)

        for idx, pred in predictions:
            all_predictions[int(idx)] = pred

    print("Total prediction: {}".format(str(len(all_predictions))))

    ordered_predictions = []
    for k in sorted(all_predictions.keys()):
        ordered_predictions.append(all_predictions[k])

    with open(os.path.join(args.input_dir, 'merged_predictions.json'), 'w') as f:
        json.dump(ordered_predictions, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    args = parser.parse_args()
    main(args)