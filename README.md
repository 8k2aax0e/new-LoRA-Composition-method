needed resources: SNI dataset(you can download it from https://github.com/allenai/natural-instructions) please split the dataset by categorys and just use its english part

1. train base lora using base_lora_trainer.py

2. get noised task question by using dataloader.trans_eval_sample, we use swap_rate=0.4

3. lora_point by yield_lora_confident.py or yield_lora_stability.py

4. select lora to yield pseudo label by selected lora using select_output_to_train.py

5. train the new combin coefficient by using unsupervised_trainer.py

6. get output of the task by yield_output.py

7. get result by using eval_result.py

Main method available now, other method will be uploaded soon.
