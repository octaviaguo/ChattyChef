import argparse
import json


class Evaluator:
    @staticmethod
    def compute_acc(hyps, refs):
        all_counter = 0
        total = 0

        acc_per_conversation = []
        for f_name in refs:
            conversation_counter = 0
            hyp_alignments = hyps[f_name]
            ref_alignments = refs[f_name]
            for hyp_alignment, ref_alignment in zip(hyp_alignments, ref_alignments):
                if hyp_alignment == ref_alignment:
                    all_counter += 1
                    conversation_counter += 1
                total += 1
            acc = conversation_counter*1./len(hyp_alignments)
            acc_per_conversation.append(acc)
        micro_acc = all_counter*1./total
        macro_acc = sum(acc_per_conversation)/len(acc_per_conversation)

        results = {
            'macro_acc': round(micro_acc, 2),
            'micro_acc': round(macro_acc, 2)
        }

        return results


def main(args):
    with open(args.hyp_file) as f:
        hyps = json.load(f)
    with open(args.ref_file) as f:
        refs = json.load(f)

    splits = ['train', 'val', 'test']
    for split in splits:
        scores = Evaluator.compute_acc(hyps[split], refs[split])
        print("{}: {}".format(split, json.dumps(scores)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--hyp_file", type=str, required=True)
    parser.add_argument("--ref_file", type=str, required=True)
    args = parser.parse_args()
    main(args)
