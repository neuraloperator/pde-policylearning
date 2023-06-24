import argparse
import yaml


def save_arguments_to_yaml(args, filepath):
    with open(filepath, 'w') as file:
        yaml.dump(vars(args), file)


def load_arguments_from_yaml(filepath):
    with open(filepath, 'r') as file:
        args_dict = yaml.safe_load(file)
    return argparse.Namespace(**args_dict)


def merge_args_with_yaml(args, yaml_args):
    # Convert Namespace object to dictionary
    args_dict = vars(args)

    # Merge YAML args into the command-line args dictionary
    args_dict.update(vars(yaml_args))

    # Convert the dictionary back to a Namespace object
    merged_args = argparse.Namespace(**args_dict)

    return merged_args


def parse_arguments():
    parser = argparse.ArgumentParser(description='Argument Controller')
    parser.add_argument('--control_yaml', type=str, default='configs/base_control.yaml',
                        help='yaml path to load configs')
    parser.add_argument('--train_yaml', type=str, default='configs/base_rno.yaml',
                        help='yaml path to load configs')
    args = parser.parse_args()
    return args
