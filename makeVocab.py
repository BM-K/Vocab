from tokenizers import BertWordPieceTokenizer

"""
with open('A_food.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    print(lines)
    exit()
"""

tokenizer = BertWordPieceTokenizer(
        vocab_file=None,
        clean_text=True,
        handle_chinese_chars=True,
        strip_accents=False,
        lowercase=False,
        wordpieces_prefix="##"
)

tokenizer.train(
        files='A_food.txt',
        limit_alphabet=6000,
        vocab_size=50265
)

tokenizer.save("./my_vocab", name='bert_vocab')

from transformers.tokenization_bert import BertTokenizer

vocab_path = "./my_vocab/bert_vocab-vocab.txt"

tokenizer = BertTokenizer(vocab_file=vocab_path, do_lower_case=False)

test_str = ' [CLS] 나는 >><< 워드피스 토크나이저를 써요. 성능이 좋은지 테스트 해보려 합니다. [SEP]'
print('테스트 문장: ',test_str)

encoded_str = tokenizer.encode(test_str, add_special_tokens=False)
print('문장 인코딩: ',encoded_str)

decoded_str = tokenizer.decode(encoded_str)
print('문장 디코딩: ',decoded_str)

