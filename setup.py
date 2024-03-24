#!/usr/bin/python3

from models.classical_approach.main.py import main as classical_main
from models.encoder_scratch_final.train_model.py import main as encoder_scratch_main
from models.bert.py import main as sota_main
def main():

    #run classical model
    classical_main()

    #run final encoder transformer from scratch  model
    encoder_scratch_main()

    #run BERT model
    sota_main()


if __name__ == "__main__":
    main()