# v1 2 this is true
# 10 OOM
python cls_with_bert.py --task_name=sentiment --do_train=true --do_eval=true \
--do_predict=true --data_dir=./data/ --vocab_file=../lib/robert/model/roberta_zh_l12/vocab.txt  \
--bert_config_file=../lib/robert/model/roberta_zh_l12/bert_config.json \
--init_checkpoint=../lib/robert/model/roberta_zh_l12/bert_model.ckpt --max_seq_length=256 \
--train_batch_size=8 --learning_rate=1e-5 --output_dir=./bert_output_robert_12L_1/ --num_train_epochs 9
