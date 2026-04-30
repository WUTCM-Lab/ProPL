# # train
# python -u code/main.py > log/semi_self_mymodel_busi_MD.log --root /home/ai1015/wqc/semi/public_data/Myocardium --mode train  --manner semi --ratio 2 --batch_size 16  --GPUs 6\
#         --dataset BUSI  --expID 1 --ckpt_name 'semi_self_mymodel_busi_MD' 2>&1 & # tn3k-1289
    # --root log/ --dataset tn3k  --expID 2   # tn3k-322
    # --root log/ --dataset tn3k  --expID 0 \  # tn3k-644

modes=("Breast_Cancer" "Fetal_Head" "Left_Atrium" "Left_Ventricle" "Myocardium" "Ovarian_Tumor" "Thyroid_Gland" "Thyroid_Nodule")

# test
for mode in "${modes[@]}"; do
    echo "Processing mode: $mode"
    python code/main.py --root /mnt/ai1015/wqc/semi/new_public_data  --mode $mode  --manner test --load_ckpt best --GPUs 2\
        --dataset bfllmott --expID 2 --ckpt_name 'mymodel_twodecoder_MCDropout_Droptimes_3_head1_4_dropout_0_3_mult_10_6_var_possi_pow_2'
done


# train bs:16