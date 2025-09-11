python collective_encoder/engine.py \
                    --datatype XTC \
                    --xtcfile ./data_generation/ala2/ala2_10ns/md.xtc \
                    --tprfile ./data_generation/ala2/ala2_10ns/md.tpr \
                    --selection "(resname ALA or resname ACE or resname NME) and not element H" \
                    --dataset GRAPH \
                    --datasize 50 --sequential \
                    --networktype "GRAPH_ENCODER" \
                    --outpath ./run_test \
                    --outfolder "test" \
                    --overwrite \
                    --nepochs 10 \
                    --lr 0.0001 \
                    --normalize_inputs \
                    --scheduler_gamma 0.1 \
                    --output_to_file \

