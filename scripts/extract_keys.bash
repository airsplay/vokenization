CUDA_VISIBLE_DEVICES=$1 python vokenization/extract_vision_keys.py \
    --image-sets vg_nococo,coco_minival,coco_nominival,coco_train,cc_valid \
    --load-dir snap/xmatching/$2
