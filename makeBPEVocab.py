from tokenizers import ByteLevelBPETokenizer

tokenizer = ByteLevelBPETokenizer()

tokenizer.train(
                files='text_eng_text.txt',
                vocab_size=50265,
                special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])

tokenizer.save('my_vocab')

