import argparse
import deepspeed
import json
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from tqdm import tqdm
from sgl_eagle.utils import load_config, build_optimizer, build_criterion
from sgl_eagle import AutoModelForCausalLM, OnlineEagleTrainer


def parse_args():
    parser = argparse.ArgumentParser(description='Train Eagle3 with online data')
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    config = load_config(args.config)

    # load draft model and base model
    target_model = AutoModelForCausalLM.from_pretrained(**config.model.target_model).eval()
    draft_model = AutoModelForCausalLM(target_model.config)
    tokenizer = AutoTokenizer.from_pretrained(config.model.target_model.pretrained_model_name_or_path)
    online_trainer = OnlineEagleTrainer(draft_model, target_model, tokenizer)

    # build dataset and dataloader
    # TODO: refactor the dataset and dataloader
    traindataset = build_dataset_rank(tokenizer, args.trainpath)
    testdataset = build_dataset_rank(tokenizer, args.testpath)
    sampler = DistributedSampler(testdataset, num_replicas=world_size, rank=global_rank, shuffle=False)
    test_loader = DataLoader(testdataset, batch_size=train_config["bs"], sampler=sampler, num_workers=4, pin_memory=True,
                            collate_fn=DataCollatorWithPadding())

    train_sampler = DistributedSampler(traindataset, num_replicas=world_size, rank=global_rank, shuffle=True)
    train_loader = DataLoader(traindataset, batch_size=train_config["bs"], sampler=train_sampler, num_workers=4,
                            pin_memory=True,
                            collate_fn=DataCollatorWithPadding())
    
    # build optimizer
    optimizer = build_optimizer(config.optimizer)

    # build
    criterion = build_criterion(config.criterion)

    for epoch in range(config.train.num_epochs):
        train_sampler.set_epoch(epoch+1)
        print(f"Now training epoch {epoch}")
        draft_model.train()
        epoch_acces = [[] for _ in range(model.length)]
        epoch_plosses = [[] for _ in range(model.length)]

        # TODO: complete the training loop, evaluation, checkpoint saving, wandb logging, etc.
        for batch_idx, data in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            output = online_trainer.step(data)

            # calculate weighted loss
            ploss_weight = [0.8 ** i for i in range(len(plosses))]
            ploss = sum([ploss_weight[i] * plosses[i] for i in range(len(plosses))])
            loss = ploss
            loss.backward()
            optimizer.step()

            epoch_acces = [epoch_acces[i] + [acces[i]] for i in range(len(acces))]
            epoch_plosses = [epoch_plosses[i] + [plosses[i].item()] for i in range(len(plosses))]

    # save the model
    draft_model.save_pretrained("./output")

if __name__ == "__main__":
    main()