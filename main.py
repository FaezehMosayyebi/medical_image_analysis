##############################################################################################
#                                                                                            #
#     coded by FaMo (faezeh.mosayyebi@gmail.com)                                             #
#     Description: To run all the pipeline run this code.                                    #
#                                                                                            #
##############################################################################################
import json
from trainer import trainer

if __name__ == "__main__":
    base_dir = "C:/Mosayyebi/hippocampus/data/preprocessed_hippocampus"
    output_dir = "C:/Mosayyebi/hippocampus/output_real_1000"

    with open(
        "C:/Mosayyebi/hippocampus/hippocampus_json_files/fold0_hippocampus_dataset.json"
    ) as json_file:
        config = json.load(json_file)

    # set the following Items
    num_layers = 3
    model_type = "complex"
    conv_type = "fast"

    model_trainer_fastcomplex = trainer(
        output_dir, base_dir, config, num_layers, model_type, conv_type
    )
    model_trainer_fastcomplex.run_training()
