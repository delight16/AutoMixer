# python -u main.py --is_training 1 --root_path ./dataset/solar/ --data_path solar_AL.txt --model_id solar --model TimeMixer --data Solar --features M --seq_len 168 --label_len 3 --pred_len 1 --e_layers 3 --d_layers 2 --factor 3 --enc_in 137 --dec_in 137 --c_out 137 --des 'Exp' --itr 1 --use_norm 0 --d_model 64 --d_ff 128 --channel_independence 0 --batch_size 32 --learning_rate 0.0001 --train_epochs 10 --patience 10 --down_sampling_layers 3 --down_sampling_method avg --down_sampling_window 2 --seed 1 --op_num 5 --gpu 2

#export CUDA_VISIBLE_DEVICES=0
# model_name=AutoCTS+
model_name=TimeMixer
is_training=2
seq_len=336
down_sampling_layers=2
down_sampling_window=2
learning_rate=0.001
batch_size=4
train_epochs=10
patience=3
gpu=0
root_path=./dataset/solar/
data_path=solar_AL.txt

python -u main.py \
  --is_training $is_training \
  --root_path $root_path \
  --data_path $data_path \
  --model_id solar_96_96 \
  --model $model_name \
  --data Solar \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 96 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp' \
  --itr 1 \
  --use_norm 0 \
  --d_model 512 \
  --d_ff 2048 \
  --channel_independence 0 \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window \
  --gpu $gpu \
  --checkpoints ./checkm4_cts+/

python -u main.py \
  --is_training $is_training \
  --root_path $root_path \
  --data_path $data_path \
  --model_id solar_96_192 \
  --model $model_name \
  --data Solar \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 192 \
  --e_layers 3 \
  --factor 3 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp' \
  --itr 1 \
  --use_norm 0 \
  --d_model 512 \
  --d_ff 2048 \
  --channel_independence 0 \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window \
  --gpu $gpu \
  --checkpoints ./checkm4_cts+/

python -u main.py \
  --is_training $is_training \
  --root_path $root_path \
  --data_path $data_path \
  --model_id solar_96_336 \
  --model $model_name \
  --data Solar \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 336 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp' \
  --itr 1 \
  --use_norm 0 \
  --d_model 512 \
  --d_ff 2048 \
  --channel_independence 0 \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window \
  --gpu $gpu \
  --checkpoints ./checkm4_cts+/

python -u main.py \
  --is_training $is_training \
  --root_path $root_path \
  --data_path $data_path \
  --model_id solar_96_720 \
  --model $model_name \
  --data Solar \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 720 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp' \
  --itr 1 \
  --use_norm 0 \
  --d_model 512 \
  --d_ff 2048 \
  --channel_independence 0 \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window \
  --gpu $gpu \
  --checkpoints ./checkm4_cts+/