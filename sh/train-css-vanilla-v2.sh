python main.py --dataset=css3d --dataset_path=../data/CSSDataset/CSS-vn-v2-segmenter.json --num_epochs=1000 \
  --model=tirg --loss=soft_triplet --comment=tirg_css-vn-v2-segmenter --n_epochs_validation=5 --loader_num_workers 6 --skip_eval_trainset True --pretrained_weights=runs/Dec19_03-02-00_ai-servers-1new-tirgphobert_css-vn-v2-segmenter/best_checkpoint.pth