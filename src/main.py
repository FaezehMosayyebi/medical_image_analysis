# How to run the model?
# Modify model config in the config.yml file
# Run this file

from trainer import trainer

if __name__ == "__main__":

    model_trainer = trainer()
    model_trainer.run_training()
