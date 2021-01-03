python main.py --dataset=mitstates --dataset_path=../data/MITStates/release_dataset \
 --model=tirg --loss=soft_triplet \
 --learning_rate_decay_frequency=50000 --num_epochs=1000  --weight_decay=5e-5 \
 --comment=mitstates_eng --skip_eval_trainset=True
