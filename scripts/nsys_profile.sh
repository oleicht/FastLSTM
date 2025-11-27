nsys profile -o smallHidden \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --force-overwrite=true \
  --duration 120 \
  python scripts/profiling.py
