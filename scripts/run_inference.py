import argparse
from sgl_eagle import AutoModelForCausalLM, EagleRunner
from transformers import AutoTokenizer



def parse_args():
    parser = argparse.ArgumentParser(description='Train Eagle3 with online data')
    parser.add_argument('-b', '--base_model_path', type=str, required=True)
    parser.add_argument('-d', '--draft_model_path', type=str, required=True)
    parser.add_argument('-p', '--prompt', type=str, required=True)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # load models
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model_path).eval()
    draft_model = AutoModelForCausalLM.from_pretrained(args.draft_model_path).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)

    # load tokenizer
    runner = EagleRunner(base_model, draft_model, tokenizer)

    # run inference
    output, usage = runner.run(args.prompt)
    print(f"The generated text text:\n{output}\n")
    print(f"The usage stats:\n{usage}")


if __name__ == "__main__":
    main()