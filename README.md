# Seq2Seq

## training
`python3 train.py
    -train_src <train_src> -train_tgt <train_tgt>
    -valid_src <valid_src> -valid_tgt <valid_tgt>`

Available options is following:
* `--vocab_size, -vs`: The number of vocabulary
* `--embed_size, -es`: The embedding size
* `--hidden_size, -es`: The hidden layer size
* `--epochs, -e`: The number of epochs
* `--batch_size, -bs`: The batch size
* `--gpu_id, -g`: GPU ID you want to use
* `--model_prefix, -mf`: The models prefix name (Encoder model is saved as `"model prefix".enc`, decoder model is saved as `"model prefix".dec`)
* `--reverse, -r`: Flag whether input sequence is reversed

train.pyを動かすと、作業ディレクトリ以下に`s_vocab.pkl`、`t_vocab.pkl`、`model.enc`、`model.dec`が作成されます。それぞれ、原言語側の辞書(語彙->ID)、目的言語側の辞書(語彙->ID)、エンコーダのモデル、デコーダのモデルです。

また、`loss_curve.pdf`が作成されます。これは各エポックでのロスの訓練データの合計、開発データでの合計がプロットされています。合計なので文の数での正規化はしてないことに注意してください。

GPUは現在1つしか指定できません。2つ以上使えるようにするかはやる気次第です。

## translate
`python3 translate.py -src <src> -output <output>`
This program is in progress. (There may be something wrong.)
