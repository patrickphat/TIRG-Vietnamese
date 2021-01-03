python main.py --dataset=css3d --dataset_path=../data/CSSDataset/CSS-vn-v2-segmenter.json --num_epochs=1000 \
  --model=tirg --loss=soft_triplet --comment=tirg_css-vn-v2-segmenter --n_epochs_validation=5 --loader_num_workers 6 --pretrained_weights="runs/Dec17_11-17-51_ai-servers-1tirg_css-vn-v2-segmenter/best_checkpoint.pth" \
  --skip_eval_trainset=True
