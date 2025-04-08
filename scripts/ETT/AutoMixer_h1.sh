model_name=AutoMixer

seq_len=336
e_layers=2
down_sampling_layers=3
down_sampling_window=2
learning_rate=0.001
d_model=16
d_ff=32
train_epochs=30
patience=5
batch_size=128

python -u runAuto.py \
  --is_training 1 \
  --root_path  ./dataset/ETT-small/\
  --data_path ETTh1.csv \
  --model_id ETTh1_$seq_len'_'96 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 96 \
  --e_layers $e_layers \
  --enc_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --batch_size $batch_size \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window

python -u runAuto.py \
  --is_training 2 \
  --root_path  ./dataset/ETT-small/\
  --data_path ETTh1.csv \
  --model_id ETTh1_$seq_len'_'96 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 96 \
  --e_layers $e_layers \
  --enc_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --batch_size $batch_size \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window



python -u runAuto.py --is_training 1 --root_path  ./dataset/ETT-small/ --data_path ETTh1.csv --model_id ETTh1_336'_'96 --model AutoMixer --data ETTh1 --features M --seq_len 336 --label_len 0 --pred_len 96 --e_layers 5 --enc_in 7 --c_out 7 --des 'Exp' --itr 1 --d_model 16 --d_ff 32 --learning_rate 0.001 --train_epochs 20 --patience 5 --batch_size 128 --down_sampling_layers 3 --down_sampling_method avg --down_sampling_window 2