nsys profile -o big \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --force-overwrite=true \
  python scripts/profiling.py
