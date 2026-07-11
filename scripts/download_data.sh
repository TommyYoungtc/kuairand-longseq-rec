#!/usr/bin/env bash
# 用法: bash scripts/download_data.sh [pure|1k|27k|meta]
set -e
mkdir -p data/raw && cd data/raw

case "${1:-pure}" in
  pure)
    wget -c https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz
    tar -xzvf KuaiRand-Pure.tar.gz ;;
  1k)
    wget -c https://zenodo.org/records/10439422/files/KuaiRand-1K.tar.gz
    tar -xzvf KuaiRand-1K.tar.gz ;;
  27k)
    wget -c https://zenodo.org/records/10439422/files/KuaiRand-27K.tar.gz
    tar -xzvf KuaiRand-27K.tar.gz ;;
  meta)
    # 视频 caption 与四级类目(第3周语义增强用)
    wget -c https://zenodo.org/records/18159199/files/kuairand_video_captions.csv
    wget -c https://zenodo.org/records/18159199/files/kuairand_video_categories.csv ;;
  *)
    echo "unknown target: $1"; exit 1 ;;
esac
echo "done."
