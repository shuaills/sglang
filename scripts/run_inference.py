import argparse
from sgl_eagle import AutoModelForCausalLM, EagleRunner
from transformers import AutoTokenizer



def parse_args():
    parser = argparse.ArgumentParser(description='Train Eagle3 with online data')
    parser.add_argument('-b', '--base_model_path', type=str, required=True)
    parser.add_argument('-d', '--draft_model_path', type=str, required=True)
    parser.add_argument('-p', '--prompt', type=str, default="List the top 10 countries in the world.")
    parser.add_argument('-e', '--eagle3', action='store_true')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # load tokenizer
    runner = EagleRunner(args.base_model_path, args.draft_model_path)

    # run inference
    output, usage = runner.run(args.prompt, enable_eagle3=args.eagle3, max_new_tokens=1024)
    print("==== Generation ====")
    print(f"Prompt:\n{args.prompt}\n\n")
    print(f"Response:\n{output[0]}\n\n")
    print(usage)


if __name__ == "__main__":
    main()