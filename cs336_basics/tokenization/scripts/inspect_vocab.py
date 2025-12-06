import pickle


def main():
    with open("training_results_ts_final/vocab.pkl", "rb") as f:
        data = pickle.load(f)

    vocab = list(data.values())

    vocab.sort(key=lambda word: len(word))

    for v in vocab[:1000]:
        print(v)

    print()
    print()
    print()

    for v in vocab[-100:]:
        print(v)


if __name__ == "__main__":
    main()
