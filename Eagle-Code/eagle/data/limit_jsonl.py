import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=70000)
    args = parser.parse_args()
    written = 0
    with open(args.input, "r", encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            if written >= args.limit:
                break
            fout.write(line)
            written += 1


if __name__ == "__main__":
    main()
