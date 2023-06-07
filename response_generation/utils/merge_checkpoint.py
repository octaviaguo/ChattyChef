import argparse
from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--saved_checkpoint", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    # lightning deepspeed has saved a directory instead of a file
    convert_zero_checkpoint_to_fp32_state_dict(args.saved_checkpoint, args.output_path)