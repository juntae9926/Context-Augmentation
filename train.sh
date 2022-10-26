python3 main.py --lr 0.001 \
                --batch-size 64 \
                --scheduler cosine \
                --criterion BCE \
                --device cuda:0 \
                --project-name context \
                --save-dir runs/no_method

python3 main.py --lr 0.001 \
                --batch-size 64 \
                --scheduler cosine \
                --criterion BCE \
                --device cuda:0 \
                --method1 \
                --project-name context \
                --save-dir runs/method1