from __future__ import print_function, division
import sys
import copy

from lang import *
from networks import *
from train_eval import *
from train_langmod import *


use_cuda = torch.cuda.is_available()

# src, tar = '../../data/hard_pc_src_syn.txt', '../../data/hard_pc_tar_syn.txt'
# src, tar = '../../data/hard_pc_src.txt', '../../data/hard_pc_tar.txt'
# src, tar = '../../data/hard_pc_src2.txt', '../../data/hard_pc_tar2.txt'
#dirpath = '/Users/romapatel/github/ltl-amdp/lggltl/'
src, tar = '../../data/hard_pc_src_syn2.txt', '../../data/hard_pc_tar_syn2.txt'
src, tar = '../../data/hard_pc_src_syn2_dup.txt', '../../data/hard_pc_tar_syn2_dup.txt'


#SEED = int(sys.argv[1])
#MODE = int(sys.argv[2])
SEED = 1; MODE = 0;
random.seed(SEED)
torch.manual_seed(SEED) if not use_cuda else torch.cuda.manual_seed(SEED)
print('Running with random seed {0}'.format(SEED))

input_lang, output_lang, pairs, MAX_LENGTH, MAX_TAR_LENGTH = prepareData(src, tar, False)
random.shuffle(pairs)
print('Maximum source sentence length: {0}'.format(MAX_LENGTH))
print(random.choice(pairs))

embed_size = 50
hidden_size = 256
encoder1 = EncoderRNN(input_lang.n_words, embed_size, hidden_size)
attn_decoder1 = AttnDecoderRNN(embed_size, hidden_size, output_lang.n_words)
new_attn_decoder1 = NewAttnDecoderRNN(embed_size, hidden_size, output_lang.n_words, MAX_LENGTH)
com_attn_decoder1 = CombinedAttnDecoderRNN(embed_size, hidden_size, output_lang.n_words, MAX_LENGTH)
decoder1 = DecoderRNN(embed_size, hidden_size, output_lang.n_words)

if use_cuda:
    encoder1 = encoder1.cuda()
    attn_decoder1 = attn_decoder1.cuda()
    new_attn_decoder1 = new_attn_decoder1.cuda()
    decoder1 = decoder1.cuda()
    com_attn_decoder1 = com_attn_decoder1.cuda()

SAVE = False
CLI = False


def main():
    #n_iters = 1000
    #n_iters = 5000
    n_iters = 8500
    #n_iters = 10000
    #n_iters = 1000000
    if MODE == 0:
        split = int(len(pairs)*0.2)
        train_pairs, test_pairs = pairs[split:], pairs[:split]
        trainIters(input_lang, output_lang, encoder1, attn_decoder1, train_pairs, n_iters, MAX_LENGTH, print_every=500)
        encoder1.eval()
        attn_decoder1.eval()
        print('\n\nEvaluating!\n')
        #evaluateRandomly(input_lang, output_lang, encoder1, attn_decoder1, pairs, MAX_LENGTH)
        evaluateSelected(input_lang, output_lang, encoder1, attn_decoder1, test_pairs, MAX_LENGTH)
        evaluateTraining(input_lang, output_lang, encoder1, attn_decoder1, test_pairs, MAX_LENGTH)

    elif MODE == 1:
        trainIters(input_lang, output_lang, encoder1, attn_decoder1, pairs, 10000, MAX_LENGTH, print_every=500)
        encoder1.eval()
        attn_decoder1.eval()
        evaluateTraining(input_lang, output_lang, encoder1, attn_decoder1, pairs, MAX_LENGTH)
    elif MODE == 2:
        print('Running cross validation on encoder and BA decoder...')
        crossValidation(input_lang, output_lang, encoder1, attn_decoder1, pairs, MAX_LENGTH)
    elif MODE == 3:
        print('Running cross validation on encoder and vanilla decoder...')
        crossValidation(input_lang, output_lang, encoder1, decoder1, pairs, MAX_LENGTH)
    elif MODE == 4:
        print('Running cross validation on encoder and EAA decoder...')
        crossValidation(input_lang, output_lang, encoder1, new_attn_decoder1, pairs, MAX_LENGTH)
    elif MODE == 5:
        print('Running generalization experiment with encoder and BA decoder...')
        results = []
        for i in reversed(range(1, 10)):
            acc = evalGeneralization(input_lang, output_lang, encoder1, attn_decoder1, pairs, 0.1 * i, MAX_LENGTH)
            results.append(acc)
            encoder1.apply(resetWeights)
            attn_decoder1.apply(resetWeights)
        print(', '.join(map(str, reversed(results))))
    elif MODE == 6:
        print('Running generalization experiment with encoder and EAA decoder...')
        results = []
        for i in reversed(range(1, 10)):
            acc = evalGeneralization(input_lang, output_lang, encoder1, new_attn_decoder1, pairs, 0.1 * i, MAX_LENGTH)
            results.append(acc)
            encoder1.apply(resetWeights)
            new_attn_decoder1.apply(resetWeights)
        print(', '.join(map(str, reversed(results))))
    elif MODE == 7:
        print('Running generalization experiment with encoder and vanilla decoder...')
        results = []
        for i in reversed(range(1, 10)):
            acc = evalGeneralization(input_lang, output_lang, encoder1, decoder1, pairs, 0.1 * i, MAX_LENGTH)
            results.append(acc)
            encoder1.apply(resetWeights)
            decoder1.apply(resetWeights)
        print(', '.join(map(str, reversed(results))))
    elif MODE == 8:
        print('Running generalization experiment with encoder and CA decoder...')
        results = []
        for i in reversed(range(1, 10)):
            acc = evalGeneralization(input_lang, output_lang, encoder1, com_attn_decoder1, pairs, 0.1 * i, MAX_LENGTH)
            results.append(acc)
            encoder1.apply(resetWeights)
            com_attn_decoder1.apply(resetWeights)
        print(', '.join(map(str, reversed(results))))

    elif MODE == 200:
        langmod_path = './langmod_pre_train.pt'
        data = '../../data/gltl_langmod2.txt'
        corpus = Corpus(data)
        batch_size = 64
        train_data = batchify(corpus.train, batch_size)
        bptt = 1
        num_epochs = 3
        langmod = Langmod(50, 256, output_lang.n_words)

        if use_cuda:
            langmod = langmod.cuda()

        if not os.path.exists(langmod_path):
            print('Pre-training RNN language model...')

            for epoch in range(num_epochs):
                langmod_train(data, langmod, batch_size, bptt, epoch, log_interval=200, lr=1.0)

            torch.save(langmod.state_dict(), langmod_path)
        else:
            langmod.load_state_dict(torch.load(langmod_path))
        orig_e = copy.deepcopy(langmod.embed.weight)
        attn_decoder1.inherit(langmod)

        print('Running generalization + pre-training experiment with encoder and BA decoder...')
        results = []
        for i in reversed(range(1, 10)):
            # acc = evalGeneralization(input_lang, output_lang, encoder1, attn_decoder1, pairs, 0.1 * i, MAX_LENGTH)
            acc = evalGeneralizationPT(input_lang, output_lang, encoder1, attn_decoder1, langmod, pairs, 0.1 * i, MAX_LENGTH,
                                     train_data, batch_size, bptt)
            results.append(acc)
            print(results)
            encoder1.apply(resetWeights)
            attn_decoder1.apply(resetWeights)
            langmod = Langmod(50, 256, output_lang.n_words)
            if use_cuda:
                langmod = langmod.cuda()

            langmod.load_state_dict(torch.load(langmod_path))
            attn_decoder1.inherit(langmod)

            assert torch.sum(attn_decoder1.embedding.weight - orig_e).data[0] == 0.0
        print(', '.join(map(str, reversed(results))))



    # elif MODE == 7:
    #     results = []
    #     for i in range(1, 10):
    #         acc = evalSampleEff(input_lang, output_lang, encoder1, attn_decoder1, pairs, 0.1 * i, MAX_LENGTH)
    #         results.append(acc)
    #         encoder1.apply(resetWeights)
    #         attn_decoder1.apply(resetWeights)
    #     print(', '.join(map(str, results)))
    elif MODE == 100:
        from flask import Flask, request, jsonify

        app = Flask(__name__)

        encoder1.load_state_dict(torch.load('./pytorch_encoder'))
        attn_decoder1.load_state_dict(torch.load('./pytorch_decoder'))

        @app.route('/model')
        def model():
            nl_command = request.args.get('command')
            output_words, _ = evaluate(input_lang, output_lang, encoder1, attn_decoder1, nl_command, MAX_LENGTH)
            return ' '.join(output_words[:-1])

        app.run()
    else:
        print('Unknown MODE specified...exiting...')
        sys.exit(0)

    if SAVE:
        print('Serializing trained model...')
        torch.save(encoder1.state_dict(), './pytorch_encoder')
        torch.save(attn_decoder1.state_dict(), './pytorch_decoder')
        print('Serialized trained model to disk...')

    if CLI:
        while True:
            try:
                input_sentence = raw_input("Enter a command: ")
                output_words, attentions = evaluate(input_lang, output_lang, encoder1, attn_decoder1, input_sentence, MAX_LENGTH)
                print('input =', input_sentence)
                print('output =', ' '.join(output_words))
            except EOFError:
                break

main()
