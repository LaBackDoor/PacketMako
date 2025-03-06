import sys
from model.train import ModelTrainer
from model.test import ModelTester

def main():
    # check correct command line in train or test mode 
    if len(sys.argv) < 4 or len(sys.argv) > 5 or (sys.argv[1] != "train" and sys.argv[1] != "test"):
        sys.stderr.write("Incorrect command line: train <config_path> <output_dir> (test <model_path> <pcap_dile_path>)\n")
        return

    mode = sys.argv[1] # train or test

    # train
    if mode == "train":
        if len(sys.argv) > 4:
            sys.stderr.write("Incorrect command line: train <config_path>\n")
            return
        model_trainer = ModelTrainer(sys.argv[2])
        model_trainer.train(output_dir=sys.argv[3])
    # test
    elif mode == "test":
        if len(sys.argv) != 5:
            sys.stderr.write("Incorrect command line: test <config_path> <model_path> <pcap_dile_path>\n")
            return
        model_tester = ModelTester(sys.argv[2])
        model_tester.test(sys.argv[3], sys.argv[4])

if __name__ == "__main__":
    main()