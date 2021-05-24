import argparse


class FedArgs:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("-s", "--square", type=int, default=1, help="display a square of a given number")
        parser.parse_args()

