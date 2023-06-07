import argparse
import os
import json

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.distributed as dist
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from transformers import (
    MODEL_MAPPING,
    AutoTokenizer
)
from data_modules.t5_data_module import DataModule
from data_modules.incontext_data_module import DataModule as IncontextDatamodule
from model import DSTModel
from datasets import disable_caching

# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

AVAIL_GPUS = torch.cuda.device_count()


def main(args):
    seed_everything(args.seed)

    dict_args = vars(args)
    if args.do_train:
        dm = DataModule(**dict_args)
        tokenizer_path = os.path.join(args.output_dir, "{}_tokenizer".format(args.model_name_or_path.split("/")[-1]))
        dm.save_tokenizer(tokenizer_path=tokenizer_path)

        dm.setup("fit")
        for batch in dm.train_dataloader():
            print(dm.tokenizer.batch_decode(batch['input_ids'][:3], skip_special_tokens=False))
            print(dm.tokenizer.batch_decode(
                torch.where(batch['labels'][:3] != -100, batch['labels'][:3], dm.tokenizer.eos_token_id),
                skip_special_tokens=False))
            break

        checkpoint_callback = ModelCheckpoint(save_top_k=1,
                                              save_last=True,
                                              monitor="perplexity",
                                              save_on_train_epoch_end=False)
        lr_monitor = LearningRateMonitor(logging_interval='step')
        vocab_size = len(dm.tokenizer)
        dict_args['vocab_size'] = vocab_size
        if args.finetune_from_checkpoint:
            lt_model = DSTModel.load_from_checkpoint(args.finetune_from_checkpoint,
                                                     init_from_pretrained=False,
                                                     model_name_or_path=args.model_name_or_path,
                                                     vocab_size=len(dm.tokenizer),
                                                     strict=False,
                                                     learning_rate=args.learning_rate)
        else:
            lt_model = DSTModel(**dict_args)

        total_batch_size = args.train_batch_size * AVAIL_GPUS
        if lt_model.global_rank == 0:
            print("***** Running training *****")
            print(f"  Num examples = {dm.get_num_samples()}")
            print(f"  Num Epochs = {args.num_train_epochs}")
            print(f"  Instantaneous batch size per device = {args.train_batch_size}")
            print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
            print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
            print(f"  Total optimization steps = {args.num_train_epochs}")

        trainer = Trainer(val_check_interval=args.val_check_interval,
                          max_epochs=args.num_train_epochs,
                          devices=AVAIL_GPUS,
                          deterministic=True,
                          default_root_dir=args.output_dir,
                          accumulate_grad_batches=args.gradient_accumulation_steps,
                          strategy=args.strategy,
                          accelerator='gpu',
                          num_nodes=args.num_nodes,
                          callbacks=[checkpoint_callback,
                                     lr_monitor,
                                     EarlyStopping(monitor="perplexity", patience=args.eval_patience, mode="min")])

        trainer.fit(lt_model, dm)
        print("----------------------------")
        print("Best checkpoint path: {}".format(checkpoint_callback.best_model_path))
        print("Best checkpoint score: {}".format(str(checkpoint_callback.best_model_score)))

    if args.do_incontext:
        dm = IncontextDatamodule(**dict_args)
        dm.setup('predict')
        predict_dataloader = dm.predict_dataloader()

        for batch in predict_dataloader:
            print(dm.tokenizer.batch_decode(batch['input_ids'][:3], skip_special_tokens=False))
            break

        lt_model = DSTModel(**dict_args)
        gen_kwargs = {'num_beams': 5,
                      'max_new_tokens': 64,
                      'pad_token_id': dm.tokenizer.eos_token_id}

        lt_model.setup_generation(gen_kwargs)
        trainer = Trainer(devices=AVAIL_GPUS,
                          strategy=args.strategy,
                          accelerator='gpu',
                          num_nodes=args.num_nodes)
        predictions = trainer.predict(lt_model, predict_dataloader)
        generated_text = dm.decode_predictions(predictions)

        current_rank = lt_model.global_rank
        tmp_file = args.prefix + "predictions_" + str(current_rank) + ".json"
        with open(os.path.join(args.output_dir, tmp_file), 'w') as f:
            json.dump(generated_text, f, indent=2)

        dist.barrier()

    if args.do_test or args.do_predict:
        dm = DataModule(**dict_args)
        if args.do_test:
            stage = 'test'
        else:
            stage = 'predict'
        dm.setup(stage)

        if args.do_test:
            predict_dataloader = dm.test_dataloader()
        else:
            predict_dataloader = dm.predict_dataloader()

        for batch in predict_dataloader:
            print(dm.tokenizer.batch_decode(batch['input_ids'][:3], skip_special_tokens=False))
            break
        lt_model = DSTModel.load_from_checkpoint(args.best_checkpoint,
                                                 init_from_pretrained=False,
                                                 model_name_or_path=args.model_name_or_path,
                                                 vocab_size=len(dm.tokenizer),
                                                 strict=False)

        gen_kwargs = {'num_beams': 5,
                      'max_length': args.max_target_length}

        lt_model.setup_generation(gen_kwargs)

        trainer = Trainer(devices=AVAIL_GPUS,
                          strategy=args.strategy,
                          accelerator='gpu',
                          num_nodes=args.num_nodes)

        predictions = trainer.predict(lt_model, predict_dataloader)
        generated_text = dm.decode_predictions(predictions)

        current_rank = lt_model.global_rank
        tmp_file = args.prefix + "predictions_" + str(current_rank) + ".json"
        with open(os.path.join(args.output_dir, tmp_file), 'w') as f:
            json.dump(generated_text, f, indent=2)

        dist.barrier()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--best_checkpoint", default=None, type=str,
                        help="The best checkpoint path.")
    parser.add_argument("--tokenizer_path", default=None, type=str, help="Path to pre-trained tokenizer")
    parser.add_argument("--prefix", default='', type=str)

    parser.add_argument('--seed', type=int, default=1234, help="random seed for initialization")
    parser.add_argument("--max_source_length", default=1024, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", default=64, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--context_window", default=1, type=int, help="Conversation history.")
    parser.add_argument("--num_train_epochs", default=15, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--num_nodes", type=str, default=1, help="Number of GPU nodes for distributed training.")
    parser.add_argument("--logging_steps", default=100, type=int)
    parser.add_argument("--saving_steps", default=100, type=int)
    parser.add_argument("--val_check_interval", default=1.0, type=float)
    parser.add_argument("--eval_patience", type=int, default=-1,
                        help="wait N times of decreasing dev score before early stop during training")

    parser.add_argument("--num_beams", type=int, default=5,
                        help="Number of beams for beam search. 1 means no beam search.")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=None,
                        help="If set to int > 0, all ngrams of that size can only occur once.")

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--finetune_from_checkpoint", type=str, default=None,
                        help="path to a previous checkpoint to continue fine-tuning on")
    parser.add_argument("--do_validation", action="store_true", help="Whether to run eval on the valiation set.")
    parser.add_argument("--do_test", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to run eval on the predict set.")
    parser.add_argument("--do_incontext", action="store_true", help="Whether to run incontext learning.")
    parser.add_argument("--train_dataset_path", type=str)
    parser.add_argument("--predict_dataset_path", type=str)
    parser.add_argument("--num_incontext_examples", type=int, default=1)
    parser.add_argument("--from_csv", type=str, default=None, help="Path to the csv file which is needed to generate")
    parser.add_argument("--out_prediction_path", type=str, default=None, help="Path to the prediction file")
    parser.add_argument("--description", type=str, choices=["cookdial", "newdataset"], help="Description mode")

    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model")
    parser.add_argument("--dataset_name", default=None, type=str, required=True, help="Dataset name")

    parser.add_argument("--strategy", default=None, type=str, help="Training strategy")
    parser.add_argument("--precision", default=32, type=str)

    parser.add_argument("--preprocessing_num_workers", default=24, type=int)
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--train_batch_size", default=128, type=int, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int, help="Batch size for evaluation.")

    args = parser.parse_args()
    if args.tokenizer_path is None:
        args.tokenizer_path = args.model_name_or_path
    disable_caching()
    main(args)
