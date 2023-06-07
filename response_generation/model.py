import torch
from pytorch_lightning import LightningModule
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoConfig,
    AdamW,
    get_linear_schedule_with_warmup
)
from datasets import load_metric


class DialogGeneration(LightningModule):
    def __init__(
            self,
            model_name_or_path: str,
            init_from_pretrained: bool = True,
            learning_rate: float = 2e-5,
            adam_epsilon: float = 1e-8,
            warmup_steps: int = 0,
            weight_decay: float = 0.0,
            train_batch_size: int = 32,
            eval_batch_size: int = 32,
            vocab_size: int = -1,
            model_class_name: str = "t5",
            max_length: int = 128,
            num_beams: int = None,
            **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

        if 'gpt' in model_name_or_path:
            auto_model = AutoModelForCausalLM
        else:
            auto_model = AutoModelForSeq2SeqLM
        config = AutoConfig.from_pretrained(model_name_or_path)
        if init_from_pretrained:
            if 'gpt-j' in model_name_or_path:
                self.model = auto_model.from_pretrained(model_name_or_path, revision="float16",
                                                        torch_dtype=torch.float16, low_cpu_mem_usage=True)
            else:
                self.model = auto_model.from_pretrained(model_name_or_path)
        else:
            self.model = auto_model.from_config(config)

        if vocab_size > -1 and config.vocab_size < vocab_size:
            self.model.resize_token_embeddings(vocab_size)

        self.gen_kwargs = {
            "max_length": max_length,
            "num_beams": num_beams,
        }

        self.gen_kwargs = None
        self.total_steps = 0

    def training_step(self, batch, batch_idx):
        batch.pop("sample_idx", None)
        batch.pop("prompt_length", None)
        outputs = self.model(**batch)
        loss = outputs[0]
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        batch.pop("sample_idx", None)
        batch.pop("prompt_length", None)
        outputs = self.model(**batch)
        val_loss = outputs[0]
        return val_loss

    def validation_step_end(self, loss_parts):
        mean_step_loss = torch.mean(loss_parts)
        return mean_step_loss

    def validation_epoch_end(self, outputs) -> None:
        mean_epoch_loss = torch.mean(torch.stack(outputs))
        perplexity = torch.exp(mean_epoch_loss)
        self.log("validation loss", mean_epoch_loss, sync_dist=True, on_epoch=True, prog_bar=True)
        self.log("perplexity", perplexity, sync_dist=True, on_epoch=True, prog_bar=True)

    def predict_step(self, batch, batch_idx):
        generated_tokens = self.model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    **self.gen_kwargs)
        return batch["sample_idx"], generated_tokens, batch["prompt_length"]

    def setup_generation(self, gen_kwargs):
        self.gen_kwargs = gen_kwargs

    def setup(self, stage=None) -> None:
        if stage != "fit":
            return
        num_samples = self.trainer.datamodule.get_num_samples()
        global_batch_size = self.trainer.num_nodes * self.trainer.gpus * self.hparams.train_batch_size \
            if self.trainer.gpus > 0 else self.hparams.train_batch_size
        self.total_steps = num_samples // global_batch_size // self.trainer.accumulate_grad_batches * self.trainer.max_epochs

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        optimizer = AdamW(self.model.parameters(), lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.total_steps
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return [optimizer], [scheduler]
