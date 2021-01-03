python main.py --dataset=mitstates_vn --dataset_path=../data/MITStates/release_dataset \
 --model=tirg --loss=soft_triplet \
 --learning_rate_decay_frequency=50000 --num_epochs=100  --weight_decay=5e-5 \
 --comment=mitstates_vn_tirg --batch_size=32 
