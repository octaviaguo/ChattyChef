import sys
sys.path.append('../')
import os
import torch
from downstream_tasks_align import load_data
from response_generation.evaluation.metrics import F1Metric, BleuMetric, JaccardMetric
import argparse
import json
from sentence_transformers import SentenceTransformer, util
from nltk.tokenize import sent_tokenize, word_tokenize


class AlignModule:
    def __init__(self, sent_embedding=False, scorer=None):
        self.model = None
        self.scorer = scorer
        if sent_embedding:
            self.model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')

    # word overlap alignment
    def align1(self, recipe_steps, conversation, alpha1=0.2, alpha2=0.3):
        current_step_idx = 0
        response_idx2step_idx = []
        for i, turn in enumerate(conversation):
            speaker = turn[0]
            text = turn[1]
            if speaker == 'Agent':
                if i == 0:
                    response_idx2step_idx.append(0)
                else:
                    candidate_step_idx = current_step_idx
                    max_score = -1
                    start_idx = 0
                    end_idx = len(recipe_steps) - 1
                    for idx in range(start_idx, end_idx + 1):
                        score = self.scorer.compute(text, sent_tokenize(recipe_steps[idx]))
                        if max_score < score:
                            max_score = score
                            candidate_step_idx = idx

                    if max_score > alpha1 and (candidate_step_idx - current_step_idx == 1):
                        current_step_idx = candidate_step_idx
                    elif max_score > alpha2:
                        current_step_idx = candidate_step_idx
                    response_idx2step_idx.append(current_step_idx)
        return response_idx2step_idx

    # SentEmbedding alignment
    def align2(self, recipe_steps, conversation, alpha1=0.2, alpha2=0.3):
        current_step_idx = 0
        response_idx2step_idx = []
        recipe_embeddings = []
        for step_idx, step in enumerate(recipe_steps):
            recipe_embeddings.append(self.model.encode(sent_tokenize(' '.join(step)), convert_to_tensor=True))
        # recipe_embeddings = model.encode(recipe_steps, convert_to_tensor=True)
        for i, turn in enumerate(conversation):
            speaker = turn[0]
            text = turn[1]
            response_embedding = self.model.encode(text, convert_to_tensor=True)
            if speaker == 'Agent':
                if i == 0:
                    response_idx2step_idx.append(0)
                else:
                    candidate_step_idx = current_step_idx
                    max_score = -1
                    start_idx = 0
                    end_idx = len(recipe_steps) - 1
                    for idx in range(start_idx, end_idx + 1):
                        score = torch.max(util.cos_sim(response_embedding, recipe_embeddings[idx]))
                        if max_score < score:
                            max_score = score
                            candidate_step_idx = idx

                    if max_score > alpha1 and (candidate_step_idx - current_step_idx == 1):
                        current_step_idx = candidate_step_idx
                    elif max_score > alpha2:
                        current_step_idx = candidate_step_idx
                    response_idx2step_idx.append(current_step_idx)
        return response_idx2step_idx


def main(args):
    scorer_mapping = {
        'f1': F1Metric,
        'bleu': BleuMetric,
        'jaccard': JaccardMetric,
        'sbert': None
    }
    sent_embedding = False
    if args.metric == 'sbert':
        sent_embedding = True
    scorer = scorer_mapping[args.metric]
    align_module = AlignModule(scorer=scorer, sent_embedding=sent_embedding)

    splits = ['train', 'val', 'test']
    alignments = {}
    for split in splits:
        f_name_list, conversations, recipes = load_data(args.input_path, split)

        alignments[split] = {}
        for f_name, conversation, recipe in zip(f_name_list, conversations, recipes):
            recipe_steps = []
            for step_idx, step in enumerate(recipe):
                recipe_steps.append(' '.join(step))
            if args.metric == 'sbert':
                alignment = align_module.align2(recipe_steps, conversation, alpha1=args.alpha1, alpha2=args.alpha2)
            else:
                alignment = align_module.align1(recipe_steps, conversation, alpha1=args.alpha1, alpha2=args.alpha2)
            alignments[split][f_name] = alignment

    with open(args.output_file, 'w') as f:
        json.dump(alignments, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--metric", type=str, choices=['f1', 'bleu', 'jaccard', 'sbert'])
    parser.add_argument("--alpha1", type=float, default=0.2)
    parser.add_argument("--alpha2", type=float, default=0.3)
    args = parser.parse_args()
    main(args)

