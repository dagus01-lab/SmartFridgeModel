import numpy as np
from tqdm import tqdm
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch
import math

from transformers import AdamW, get_linear_schedule_with_warmup
KD_loss = nn.KLDivLoss(reduction='batchmean')

def generate_recipe(prompt, model, tokenizer, device):
    input = f"<|startoftext|>Prompt: {prompt.strip()}"#\nTitle:" should be added at the end if no other ingredient should be added to the list
    input = tokenizer(input, return_tensors="pt")
    input_ids = input["input_ids"]
    attention_mask = input["attention_mask"]

    output = model.generate(input_ids.to(device),
                            attention_mask=attention_mask.to(device),
                            num_beams=5,
                            max_length=768,
                            num_return_sequences=1,
                            eos_token_id=tokenizer.eos_token_id,
                            do_sample = True, top_k = 100, top_p = 0.85)
    output = tokenizer.decode(output[0], skip_special_tokens=True)
    return output

def train_distilled(teacher, student, train_dataset, val_dataset,
                epochs, batch_size, device,
                sample_every=1000, save_every=5000, save_file='model',
                learning_rate=5e-4, warmup_steps=1e2, epsilon=1e-8, temperature=2):
    # Create the DataLoaders for our training and validation datasets.
    # We'll take training samples in random order.
    train_dataloader = DataLoader(
        train_dataset,  # The training samples.
        sampler=RandomSampler(train_dataset),  # Select batches randomly
        batch_size=batch_size  # Trains with this batch size.
    )

    # For validation the order doesn't matter, so we'll just read them sequentially.
    validation_dataloader = DataLoader(
        val_dataset,  # The validation samples.
        sampler=SequentialSampler(val_dataset),  # Pull out batches sequentially.
        batch_size=batch_size  # Evaluate with this batch size.
    )

    # Note: AdamW is a class from the huggingface library (as opposed to pytorch)
    optimizer = torch.optim.AdamW(student.parameters(),
                                    lr=learning_rate,
                                    eps=epsilon
                                    )

    # Total number of training steps is [number of batches] x [number of epochs].
    # (Note that this is not the same as the number of training samples).
    total_steps = len(train_dataloader) * epochs
    print('Total number of steps: ', total_steps)
    # Create the learning rate scheduler.
    # This changes the learning rate as the training loop progresses
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)

    training_stats = []
    print("Currently using device type: ", device)

    student = student.to(device)
    teacher = teacher.to(device)

    for epoch_i in range(0, epochs):

        # ========================================
        #               Training
        # ========================================

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        losses = []

        total_train_loss = 0

        student.train()
        teacher.eval()

        for step, batch in enumerate(train_dataloader):

            b_input_ids = batch[0].to(device)
            b_labels = batch[0].to(device)
            b_masks = batch[1].to(device)

            with torch.no_grad():
                logits_t = teacher(b_input_ids,
                            labels=b_labels,
                            attention_mask=b_masks,
                            token_type_ids=None
                            )
            logits_s = student(b_input_ids,
                            labels=b_labels,
                            attention_mask=b_masks,
                            token_type_ids=None)

            loss = KD_loss(input=torch.nn.functional.log_softmax(logits_s.logits / temperature, dim=1),
                           target=torch.nn.functional.softmax(logits_t.logits / temperature, dim=1))
            ##call the method above and add student&teacher model

            batch_loss = loss.item()
            total_train_loss += batch_loss
            losses.append(batch_loss)
            perplexity = torch.exp(loss)

            # Get sample every x batches.
            if step % sample_every == 0 and not step == 0:
                print('Batch {:>5,}  of  {:>5,}. Loss: {:>5,}. Perplexity: {:>5}.'.format(step, len(train_dataloader), batch_loss, perplexity.item()))

            loss.backward()

            optimizer.step()

            scheduler.step()

            if step % save_every == 0:
                student.save_pretrained(save_file)

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)

        # Calculate perplexity.
        losses = torch.tensor(losses)
        train_perplexity = np.exp(torch.mean(losses))

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Perplexity: {0:.2f}".format(train_perplexity))
        # ========================================
        #               Validation
        # ========================================

        print("")
        print("Running Validation...")

        student.eval()

        losses = []
        total_eval_loss = 0
        nb_eval_steps = 0

        # Evaluate data for one epoch
        for batch in validation_dataloader:
            b_input_ids = batch[0].to(device)
            b_labels = batch[0].to(device)
            b_masks = batch[1].to(device)

            with torch.no_grad():
                with torch.no_grad():
                    logits_t = teacher(b_input_ids,
                                       labels=b_labels,
                                       attention_mask=b_masks,
                                       token_type_ids=None
                                       )
                    logits_s = student(b_input_ids,
                                        labels=b_labels,
                                        attention_mask=b_masks,
                                        token_type_ids=None)

                    loss = KD_loss(input=torch.nn.functional.log_softmax(logits_s.logits / temperature, dim=1),
                                    target=torch.nn.functional.softmax(logits_t.logits / temperature, dim=1))

            batch_loss = loss.item()
            losses.append(batch_loss)
            total_eval_loss += batch_loss

        avg_val_loss = total_eval_loss / len(validation_dataloader)

        # Calculate perplexity.
        losses = torch.tensor(losses)
        val_perplexity = np.exp(torch.mean(losses))

        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation perplexity: {0:.2f}".format(val_perplexity))

        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Training Perplexity': train_perplexity,
                'Valid. Perplexity': val_perplexity,
            }
        )

    print("")
    print("Training complete!")
    return training_stats

def train_model(model, train_dataset, val_dataset,
                epochs, batch_size, device,
                sample_every=1000, save_every=5000, save_file='teacher_model',
                learning_rate=5e-4, warmup_steps=1e2, epsilon=1e-8):
    # Create the DataLoaders for our training and validation datasets.
    # We'll take training samples in random order.
    train_dataloader = DataLoader(
        train_dataset,  # The training samples.
        sampler=RandomSampler(train_dataset),  # Select batches randomly
        batch_size=batch_size  # Trains with this batch size.
    )

    # For validation the order doesn't matter, so we'll just read them sequentially.
    validation_dataloader = DataLoader(
        val_dataset,  # The validation samples.
        sampler=SequentialSampler(val_dataset),  # Pull out batches sequentially.
        batch_size=batch_size  # Evaluate with this batch size.
    )

    # Note: AdamW is a class from the huggingface library (as opposed to pytorch)
    optimizer = torch.optim.AdamW(model.parameters(),
                      lr=learning_rate,
                      eps=epsilon
                      )

    # Total number of training steps is [number of batches] x [number of epochs].
    # (Note that this is not the same as the number of training samples).
    total_steps = len(train_dataloader) * epochs
    print('Total number of steps: ', total_steps)
    # Create the learning rate scheduler.
    # This changes the learning rate as the training loop progresses
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)

    training_stats = []
    print("Currently using device type: ", device)

    model = model.to(device)

    for epoch_i in range(0, epochs):

        # ========================================
        #               Training
        # ========================================
        """
        if epoch_i==0:
            print("======== Epoch 1 / 2 ========\n"\
"Training...\n"\

"Batch: 54088, Loss: 0.5402792096138: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 54089/54089 [9:47:50<00:00,  1.53it/s]"\


"  Average training loss: 0.55\n"\
"  Perplexity: 1.73\n"\

"Running Validation...\n"\

"Validation Loss: 0.4237533509731293: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6010/6010 [21:49<00:00,  4.59it/s]"\

"  Validation Loss: 0.48\n"\
"  Validation perplexity: 1.61\n")
            training_stats.append(
                {
                    'epoch': epoch_i + 1,
                    'Training Loss': 0.55,
                    'Valid. Loss': 0.48,
                    'Training Perplexity': np.exp(0.55).item(),
                    'Valid. Perplexity': np.exp(0.48).item(),
                }
            )
            continue"""
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        losses = []

        total_train_loss = 0

        model.train()
        with tqdm(total=len(train_dataloader)) as pbar:
            for step, batch in enumerate(train_dataloader):

                b_input_ids = batch[0].to(device)
                b_labels = batch[0].to(device)
                b_masks = batch[1].to(device)

                model.zero_grad()

                outputs = model(b_input_ids,
                                labels=b_labels,
                                attention_mask=b_masks,
                                token_type_ids=None
                                )

                loss = outputs[0]
                ##call the method above and add student&teacher model

                batch_loss = loss.item()
                total_train_loss += batch_loss
                losses.append(batch_loss)

                """
                # Get sample every x batches.
                if step % sample_every == 0 and not step == 0:
                    print('Batch {:>5,}  of  {:>5,}. Loss: {:>5,}.'.format(step, len(train_dataloader), batch_loss))"""

                loss.backward()

                optimizer.step()

                scheduler.step()

                if step % save_every == 0:
                    model.save_pretrained(save_file)

                pbar.set_description(f"Batch: {step}, Loss: {batch_loss}")
                pbar.update(1)

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)

        # Calculate perplexity.
        losses = torch.tensor(losses)
        train_perplexity = np.exp(torch.mean(losses))

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Perplexity: {0:.2f}".format(train_perplexity))
        torch.cuda.empty_cache()  # clear GPU memory at the end of the training epoch
        # ========================================
        #               Validation
        # ========================================

        print("")
        print("Running Validation...")

        model.eval()

        losses = []
        total_eval_loss = 0
        nb_eval_steps = 0

        # Evaluate data for one epoch
        with tqdm(total=len(validation_dataloader)) as pbar:
            for batch in validation_dataloader:
                b_input_ids = batch[0].to(device)
                b_labels = batch[0].to(device)
                b_masks = batch[1].to(device)

                with torch.no_grad():
                    outputs = model(b_input_ids,
                                    #                            token_type_ids=None,
                                    attention_mask=b_masks,
                                    labels=b_labels)

                    loss = outputs[0]

                batch_loss = loss.item()
                losses.append(batch_loss)
                total_eval_loss += batch_loss
                pbar.set_description(f"Validation Loss: {batch_loss}")
                pbar.update(1)

        avg_val_loss = total_eval_loss / len(validation_dataloader)

        # Calculate perplexity.
        losses = torch.tensor(losses)
        val_perplexity = np.exp(torch.mean(losses))

        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation perplexity: {0:.2f}".format(val_perplexity))

        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Training Perplexity': train_perplexity.item(),
                'Valid. Perplexity': val_perplexity.item(),
            }
        )

        torch.cuda.empty_cache() #clear GPU memory at the end of the epoch

    print("")
    print("Training complete!")
    return training_stats


def evaluate_model(model, test_dataset, batch_size, device):
    dataloader = DataLoader(
        test_dataset,  # The validation samples.
        sampler=SequentialSampler(test_dataset),  # Pull out batches sequentially.
        batch_size=batch_size  # Evaluate with this batch size.
    )
    model = model.to(device)
    model.eval()

    losses = []
    perplexity = []
    total_eval_loss = 0
    nb_eval_steps = 0

    # Evaluate data for one epoch
    with tqdm(total=len(dataloader)) as pbar:
        for batch in dataloader:

            b_input_ids = batch[0].to(device)
            b_labels = batch[0].to(device)
            b_masks = batch[1].to(device)

            with torch.no_grad():

                outputs = model(b_input_ids,
        #                       token_type_ids=None,
                                attention_mask = b_masks,
                                labels=b_labels)

                loss = outputs[0]

            batch_loss = loss.item()
            losses.append(batch_loss)
            total_eval_loss += batch_loss
            pbar.set_description(f"Validation Loss: {batch_loss}")
            pbar.update(1)

    avg_val_loss = total_eval_loss / len(dataloader)

    # Calculate perplexity.
    losses = torch.tensor(losses)
    val_perplexity = np.exp(torch.mean(losses))
    perplexity.append(val_perplexity)

    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation perplexity: {0:.2f}".format(val_perplexity))
    return avg_val_loss, val_perplexity

