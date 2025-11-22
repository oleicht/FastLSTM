ncu --set full \
    --target-processes all \
    --clock-control base \
    --kernel-name "triton_lstm_full_fwd_kernel" \
    --export first_pass.ncu-rep --force-overwrite \
    python scripts/profiling.py