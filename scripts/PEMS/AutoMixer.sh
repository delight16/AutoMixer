# python -u runAuto.py --is_training 1 --root_path ./dataset/PEMS/ --data_path PEMS03.npz --model_id PEMS03 --checkpoints ./checkpoint/ --model AutoMixer --data PEMS --features M --seq_len 96 --label_len 0 --pred_len 12 --e_layers 5 --d_layers 1 --factor 3 --enc_in 358 --dec_in 358 --c_out 358 --des 'Exp' --itr 1 --use_norm 0  --channel_independence 0 --d_model 128 --d_ff 256 --batch_size 64 --learning_rate 0.0005 --train_epochs 20 --patience 10 --down_sampling_layers 2 --down_sampling_method avg --down_sampling_window 2 --gpu 2 --op_num 5 --seed 42


# python -u runAuto.py --is_training 1 --root_path ./dataset/PEMS/ --checkpoints ./check08/ --data_path PEMS08.npz --model_id PEMS08 --model AutoMixer --data PEMS --features M --seq_len 96 --label_len 0 --pred_len 12 --e_layers 4 --d_layers 1 --factor 3 --enc_in 170 --dec_in 170 --c_out 170 --des 'Exp' --itr 1 --use_norm 0  --channel_independence 0 --d_model 128 --d_ff 256 --batch_size 128 --learning_rate 0.002 --train_epochs 25 --patience 10 --down_sampling_layers 2 --down_sampling_method avg --down_sampling_window 2 --gpu 4 --op_num 3 --seed 42


model_name=AutoMixer

seq_len=96
pred_len=12
down_sampling_layers=1
down_sampling_window=2
learning_rate=0.001
d_model=128
d_ff=256
batch_size=16
train_epochs=20
patience=5

python -u runAuto.py \
 --is_training 1 \
 --root_path ./dataset/PEMS/ \
 --data_path PEMS03.npz \
 --model_id PEMS03 \
 --model $model_name \
 --data PEMS \
 --features M \
 --seq_len $seq_len \
 --label_len 0 \
 --pred_len $pred_len \
 --e_layers 5 \
 --d_layers 1 \
 --factor 3 \
 --enc_in 358 \
 --dec_in 358 \
 --c_out 358 \
 --des 'Exp' \
 --itr 1 \
 --use_norm 0 \
  --channel_independence 0 \
 --d_model $d_model \
 --d_ff $d_ff \
 --batch_size 32 \
 --learning_rate $learning_rate \
 --train_epochs $train_epochs \
 --patience $patience \
 --down_sampling_layers $down_sampling_layers \
 --down_sampling_method avg \
 --down_sampling_window $down_sampling_window


python -u runAuto.py \
 --is_training 2 \
 --root_path ./dataset/PEMS/ \
 --data_path PEMS03.npz \
 --model_id PEMS03 \
 --model $model_name \
 --data PEMS \
 --features M \
 --seq_len $seq_len \
 --label_len 0 \
 --pred_len $pred_len \
 --e_layers 5 \
 --d_layers 1 \
 --factor 3 \
 --enc_in 358 \
 --dec_in 358 \
 --c_out 358 \
 --des 'Exp' \
 --itr 1 \
 --use_norm 0 \
  --channel_independence 0 \
 --d_model $d_model \
 --d_ff $d_ff \
 --batch_size 32 \
 --learning_rate $learning_rate \
 --train_epochs $train_epochs \
 --patience $patience \
 --down_sampling_layers $down_sampling_layers \
 --down_sampling_method avg \
 --down_sampling_window $down_sampling_window


python -u runAuto.py \
 --is_training 1 \
 --root_path ./dataset/PEMS/ \
 --data_path PEMS04.npz \
 --model_id PEMS04 \
 --model $model_name \
 --data PEMS \
 --features M \
 --seq_len $seq_len \
 --label_len 0 \
 --pred_len $pred_len \
 --e_layers 5 \
 --d_layers 1 \
 --factor 3 \
 --enc_in 307 \
 --dec_in 307 \
 --c_out 307 \
 --des 'Exp' \
 --itr 1 \
 --use_norm 0 \
 --channel_independence 0 \
 --d_model $d_model \
 --d_ff $d_ff \
 --batch_size 32 \
 --learning_rate $learning_rate \
 --train_epochs $train_epochs \
 --patience $patience \
 --down_sampling_layers $down_sampling_layers \
 --down_sampling_method avg \
 --down_sampling_window $down_sampling_window

python -u runAuto.py \
 --is_training 2 \
 --root_path ./dataset/PEMS/ \
 --data_path PEMS04.npz \
 --model_id PEMS04 \
 --model $model_name \
 --data PEMS \
 --features M \
 --seq_len $seq_len \
 --label_len 0 \
 --pred_len $pred_len \
 --e_layers 5 \
 --d_layers 1 \
 --factor 3 \
 --enc_in 307 \
 --dec_in 307 \
 --c_out 307 \
 --des 'Exp' \
 --itr 1 \
 --use_norm 0 \
 --channel_independence 0 \
 --d_model $d_model \
 --d_ff $d_ff \
 --batch_size 32 \
 --learning_rate $learning_rate \
 --train_epochs $train_epochs \
 --patience $patience \
 --down_sampling_layers $down_sampling_layers \
 --down_sampling_method avg \
 --down_sampling_window $down_sampling_window



python -u runAuto.py \
 --is_training 1 \
 --root_path ./dataset/PEMS/ \
 --data_path PEMS07.npz \
 --model_id PEMS07 \
 --model $model_name \
 --data PEMS \
 --features M \
 --seq_len $seq_len \
 --label_len 0 \
 --pred_len $pred_len \
 --e_layers 5 \
 --d_layers 1 \
 --factor 3 \
 --enc_in 883 \
 --dec_in 883 \
 --c_out 883 \
 --des 'Exp' \
 --itr 1 \
 --use_norm 0 \
 --channel_independence 0 \
 --d_model 256 \
 --d_ff 512 \
 --batch_size 32 \
 --learning_rate $learning_rate \
 --train_epochs $train_epochs \
 --patience $patience \
 --down_sampling_layers $down_sampling_layers \
 --down_sampling_method avg \
 --down_sampling_window $down_sampling_window


python -u runAuto.py \
 --is_training 2 \
 --root_path ./dataset/PEMS/ \
 --data_path PEMS07.npz \
 --model_id PEMS07 \
 --model $model_name \
 --data PEMS \
 --features M \
 --seq_len $seq_len \
 --label_len 0 \
 --pred_len $pred_len \
 --e_layers 5 \
 --d_layers 1 \
 --factor 3 \
 --enc_in 883 \
 --dec_in 883 \
 --c_out 883 \
 --des 'Exp' \
 --itr 1 \
 --use_norm 0 \
 --channel_independence 0 \
 --d_model 256 \
 --d_ff 512 \
 --batch_size 32 \
 --learning_rate $learning_rate \
 --train_epochs $train_epochs \
 --patience $patience \
 --down_sampling_layers $down_sampling_layers \
 --down_sampling_method avg \
 --down_sampling_window $down_sampling_window

python -u runAuto.py \
  --is_training 1 \
  --root_path ./dataset/PEMS/ \
  --data_path PEMS08.npz \
  --model_id PEMS08 \
  --model $model_name \
  --data PEMS \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len $pred_len \
  --e_layers 5 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 170 \
  --dec_in 170 \
  --c_out 170 \
  --des 'Exp' \
  --itr 1 \
  --use_norm 0 \
  --channel_independence 0 \
  --d_model 64 \
  --d_ff 128 \
  --batch_size 32 \
  --learning_rate $learning_rate \
  --train_epochs 20 \
  --patience $patience \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window

python -u runAuto.py \
  --is_training 2 \
  --root_path ./dataset/PEMS/ \
  --data_path PEMS08.npz \
  --model_id PEMS08 \
  --model $model_name \
  --data PEMS \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len $pred_len \
  --e_layers 5 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 170 \
  --dec_in 170 \
  --c_out 170 \
  --des 'Exp' \
  --itr 1 \
  --use_norm 0 \
  --channel_independence 0 \
  --d_model 64 \
  --d_ff 128 \
  --batch_size 32 \
  --learning_rate $learning_rate \
  --train_epochs 20 \
  --patience $patience \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window
