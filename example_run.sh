python engine.py    \
                    --runtype COLVAR \
                    --colvarfile $DATA_DIR/20221016_COLLLECTIVE_ENCODER_TRAINING_DATA_OAH/INPUTS \
                    --outpath . \
                    --save_model \
                    --nepochs 10 \
                    --labels "dist_hg.z" \
                    --overwrite \
                    --normalize \
                    --beta=1.0

