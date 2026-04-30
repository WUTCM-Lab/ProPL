# # train
python -u code/main.py > log/expID2/mymodel_twodecoder_MCDropout_Droptimes_4_head1_4_dropout_0_3_mult_1_6_var_possi_pow_2_del.log --root /mnt/ai1015/wqc/semi/new_public_data --mode train  --manner semi --ratio 2 --batch_size 8  --GPUs 1\
        --dataset bfllmott --drop_times 3 --pow 2 --expID 2 --ckpt_name 'mymodel_twodecoder_MCDropout_Droptimes_4_head1_4_dropout_0_3_mult_1_6_var_possi_pow_2_del' 2>&1 & # tn3k-1289
# python -u code/main.py --root /mnt/ai1015/wqc/semi/new_public_data --mode train  --manner semi --ratio 2 --batch_size 16  --GPUs 0\
#         --dataset bfllmott --drop_times 3 --pow 2 --expID 2 --ckpt_name 'mymodel_twodecoder_MCDropout_Droptimes_4_head1_4_dropout_0_3_mult_10_6_var_possi_pow_2_del' # tn3k-1289



# train bs:16
