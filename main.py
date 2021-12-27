from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
import os
import model_handling
import metric_handling
import data_handling
import debug_cross_attention

debug_cross_attention.is_debug = False

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

if __name__ == "__main__":
    # init model
    model, tokenizer = model_handling.init_model()

    # init data
    split_datasets = data_handling.init_data()

    tokenized_datasets = split_datasets.map(
        data_handling.preprocess_function,
        batched=True,
        batch_size=4,
        num_proc=4,
        remove_columns=split_datasets["train"].column_names,
        cache_file_names={"train": "./cache/train_datasets.arrow", "test": "./cache/test_datasets.arrow"}
    )

    data_collator = data_handling.DataCollatorForNormSeq2Seq(tokenizer, model=model)

    # set training arguments - these params are not really tuned, feel free to change
    num_epochs = 50
    checkpoint_path = "./oov_checkpoints"
    batch_size = 4  # change to 48 for full training
    training_args = Seq2SeqTrainingArguments(
        output_dir=checkpoint_path,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy="epoch",
        # evaluation_strategy="steps",
        save_strategy="epoch",
        gradient_accumulation_steps=1,
        predict_with_generate=True,
        save_total_limit=5,
        do_train=True,
        do_eval=True,
        logging_steps=200,  # set to 2000 for full training
        # save_steps=1000,  # set to 500 for full training
        # eval_steps=1000,  # set to 7500 for full training
        # warmup_steps=1,  # set to 3000 for full training
        # max_steps=16, # delete for full training
        num_train_epochs=num_epochs,  # uncomment for full training
        warmup_ratio=1 / num_epochs,
        logging_dir=os.path.join(checkpoint_path, 'log'),
        overwrite_output_dir=True,
        metric_for_best_model='wer',
        greater_is_better=False,
        # metric_for_best_model='bleu',
        # greater_is_better=True,
        eval_accumulation_steps=10,
        dataloader_num_workers=2,  # 20 for full training
        # sharded_ddp="simple",
        # local_rank=2,
        # fp16=True,
    )

    # instantiate trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        compute_metrics=metric_handling.get_wer_metric_compute_fn(tokenizer),
        train_dataset=tokenized_datasets['train'].shard(100, 0),
        eval_dataset=tokenized_datasets['test'].shard(100, 0),
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    # trainer.evaluate()
    # trainer.save_model(checkpoint_path)
    trainer.train()