from __future__ import print_function, division
import random
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim

import itertools

from utils import *
from train_langmod import *

SOS_token = 0
EOS_token = 1
UNK_token = 2

use_cuda = torch.cuda.is_available()


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def variableFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    if use_cuda:
        return result.cuda()
    else:
        return result


def variablesFromPair(input_lang, output_lang, pair):
    input_variable = variableFromSentence(input_lang, ' '.join(list(reversed(pair[0].split()))))
    target_variable = variableFromSentence(output_lang, pair[1])
    return input_variable, target_variable

teacher_forcing_ratio = 0.5


def train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_variable[di])
            decoder_input = target_variable[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]
            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

            loss += criterion(decoder_output, target_variable[di])
            if ni == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length
    return loss.data[0] / target_length


def trainIters(in_lang, out_lang, encoder, decoder, samples, n_iters, max_length, print_every=1000, plot_every=100, learning_rate=0.001):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    training_pairs = itertools.cycle(iter([variablesFromPair(in_lang, out_lang, s) for s in samples]))
    criterion = nn.NLLLoss()

    i, max = -1, n_iters + 1
    for i in range(1, n_iters + 1):
    #for training_pair in training_pairs:

        #i += 1
        #if i == 500: break
        #training_pair = training_pairs.next()
        training_pair = next(training_pairs)

        input_variable = training_pair[0]
        target_variable = training_pair[1]

        loss = train(input_variable, target_variable, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion, max_length)
        print_loss_total += loss
        plot_loss_total += loss


        print(i, loss)
        if i % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            #print('%s (%d %d%%) %.4f' % (timeSince(start, i / n_iters),
                                         #i, i / n_iters * 100, print_loss_avg))

        if i % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0


    #encoder1, attn_decoder1, new_attn_decoder1, com_attn_decoder1, decoder1

    import os
    path = '../../models/torch/checkpoint/'
    #path = '../../models/torch/checkpoint-gpu/'

    #path = '../../models/torch/checkpoint' + str(n_iters) + '/'
    if os.path.isdir(path) is False: os.mkdir(path)
    torch.save(encoder.state_dict(), path + 'encoder')
    torch.save(decoder.state_dict(), path + 'decoder')


def evaluate(input_lang, output_lang, encoder, decoder, sentence, max_length):
    #print('Sentence, ', sentence)
    input_variable = variableFromSentence(input_lang, ' '.join(list(reversed(sentence.split()))))
    input_length = input_variable.size()[0]
    encoder_hidden = encoder.initHidden()

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei],
                                                 encoder_hidden)
        encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))  # SOS
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)

    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs)
        if decoder_attention is not None:
            decoder_attentions[di] = decoder_attention.data

        topv, topi = decoder_output.data.topk(1)
        #ni = topi[0][0].data.numpy().tolist()
        ni = topi[0][0].cpu().data.numpy().tolist()

        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(output_lang.index2word[ni])

        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    return decoded_words, decoder_attentions[:di + 1]


def evaluate2(input_lang, output_lang, encoder, decoder, sentence, max_length):
    input_variable = variableFromSentence(input_lang, ' '.join(list(reversed(sentence.split()))))
    input_length = input_variable.size()[0]
    encoder_hidden = encoder.initHidden()

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei],
                                                 encoder_hidden)
        encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]

    decoder_hidden = encoder_hidden

    candidates = [([SOS_token], decoder_hidden, 0.0)]
    beam_width = 10
    completed = []

    while len(candidates) > 0:
        new_candidates = []
        for i in range(len(candidates)):
            seq, hidden, score = candidates[i]

            if seq[-1] == EOS_token or len(seq) == max_length:
                continue

            decoder_input = Variable(torch.LongTensor([[seq[-1]]]))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, hidden, encoder_outputs)

            topv, topi = decoder_output.data.topk(decoder_output.size()[-1])
            # print(('Input token: ', output_lang.index2word[seq[-1]]))
            # print(topi[0].numpy())
            for v, i in zip(topv[0], topi[0]):
                new_candidates.append((seq + [i], decoder_hidden, score + v))
        candidates = sorted(new_candidates, key=lambda x: x[2], reverse=True)
        # print([([output_lang.index2word[i] for i in x], z) for x, y, z in candidates])
        # print()

        candidates = candidates[:beam_width]
        next_candidates = []
        for x, y, z in candidates:
            if x[-1] == EOS_token:
                if len(completed) == 0:
                    completed.append((x, z))
                else:
                    if z > completed[0][1]:
                        completed = [(x, z)]
            else:
                next_candidates.append((x, y, z))

        candidates = next_candidates
        # print([([output_lang.index2word[i] for i in x], z) for x, y, z in candidates])
        # print()
        # print([([output_lang.index2word[i] for i in x[0]], x[1]) for x in completed])
        # raw_input()

    completed = sorted(completed, key=lambda x: x[1], reverse=True)
    decoded_words = [output_lang.index2word[i] for i in completed[0][0]]

    return decoded_words[1:], None


def evaluateSelected(input_lang, output_lang, encoder, decoder, pairs, max_length, n=10):
    n = 10
    n = len(pairs)
    items = []
    for i in range(n):
        pair = random.choice(pairs)
        #pair = pairs[i]
        print('>', pair[0])
        print('=', pair[1])

        sentence = pair[0]
        #sentence = 'move move move '
        output_words, attentions = evaluate(input_lang, output_lang, encoder, decoder, sentence, max_length)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')
        items.append(output_sentence)

    return items

def evaluateRandomly(input_lang, output_lang, encoder, decoder, pairs, max_length, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(input_lang, output_lang, encoder, decoder, pair[0], max_length)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


def evaluateTraining(input_lang, output_lang, encoder, decoder, pairs, max_length):
    corr, tot = 0, 0
    for p in pairs:
        output_words, attentions = evaluate(input_lang, output_lang, encoder, decoder, p[0], max_length)
        output_words = ' '.join(output_words[:-1])
        if output_words == p[1]:
            corr += 1
        # print('Input: {0}\tOutput: {1}\tExpected:{2}'.format(p[0], output_words, p[1]))
        tot += 1

    print('Training Accuracy: {0}/{1} = {2}%'.format(corr, tot, corr / tot))


def evaluateSamples(input_lang, output_lang, encoder, decoder, samples, max_length):
    corr, tot = 0, 0
    for p in samples:
        output_words, attentions = evaluate(input_lang, output_lang, encoder, decoder, p[0], max_length)
        output_words = ' '.join(output_words[:-1])
        # print((output_words, p[1]))
        if output_words == p[1]:
            corr += 1
        tot += 1
    return corr, tot, corr / tot


def resetWeights(m):
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()


def crossValidation(in_lang, out_lang, encoder, decoder, samples, max_length, n_folds=5):
    for _ in range(10):
        random.shuffle(samples)

    correct, total = 0, 0
    fold_range = range(0, len(samples), int(len(samples) / n_folds))
    fold_range.append(len(samples))

    print('Starting {0}-fold cross validation'.format(n_folds))
    for f in xrange(n_folds):
        print('Running cross validation fold {0}/{1}...'.format(f + 1, n_folds))

        encoder.train()
        decoder.train()

        train_samples = samples[:fold_range[f]] + samples[fold_range[f + 1]:]
        val_samples = samples[fold_range[f]:fold_range[f + 1]]

        trainIters(in_lang, out_lang, encoder, decoder, train_samples, 10000, max_length, print_every=500)

        encoder.eval()
        decoder.eval()

        corr, tot, acc = evaluateSamples(in_lang, out_lang, encoder, decoder, val_samples, max_length)
        print('Cross validation fold #{0} Accuracy: {1}/{2} = {3}%'.format(f + 1, corr, tot, 100. * acc))
        correct += corr
        total += tot

        encoder.apply(resetWeights)
        decoder.apply(resetWeights)
    print('{0}-fold Cross Validation Accuracy: {1}/{2} = {3}%'.format(n_folds, correct, total, 100. * correct / total))


def evalGeneralization(in_lang, out_lang, encoder, decoder, samples, perc, max_length):
    for _ in range(10):
        random.shuffle(samples)

    encoder.train()
    decoder.train()

    tar_set = list(set([s[1] for s in samples]))
    tar_num = int(np.ceil(perc * len(tar_set)))
    train_forms = random.sample(tar_set, tar_num)
    print('GLTL Training Formulas: {0}'.format(train_forms))
    print('GLTL Evaluation Formulas: {0}'.format([s for s in tar_set if s not in train_forms]))
    train_samples = [s for s in samples if s[1] in train_forms]
    eval_samples = [s for s in samples if s[1] not in train_forms]

    print('Training with {0}/{3} unique GLTL formulas => {1} training samples | {2} testing samples'.format(tar_num, len(train_samples), len(eval_samples), len(tar_set)))
    trainIters(in_lang, out_lang, encoder, decoder, train_samples, 10000, max_length, print_every=10000)

    encoder.eval()
    decoder.eval()

    corr, tot, acc = evaluateSamples(in_lang, out_lang, encoder, decoder, eval_samples, max_length)
    print('Held-out Accuracy: {0}/{1} = {2}%'.format(corr, tot, 100. * acc))
    return acc


def evalGeneralizationPT(in_lang, out_lang, encoder, decoder, langmod, samples, perc, max_length, train_data, batch_size, bptt):
    for _ in range(10):
        random.shuffle(samples)

    encoder.train()
    decoder.train()
    langmod.train()

    tar_set = list(set([s[1] for s in samples]))
    tar_num = int(np.ceil(perc * len(tar_set)))
    train_forms = random.sample(tar_set, tar_num)
    print('GLTL Training Formulas: {0}'.format(train_forms))
    print('GLTL Evaluation Formulas: {0}'.format([s for s in tar_set if s not in train_forms]))
    train_samples = [s for s in samples if s[1] in train_forms]
    eval_samples = [s for s in samples if s[1] not in train_forms]

    print('Training with {0}/{3} unique GLTL formulas => {1} training samples | {2} testing samples'.format(tar_num, len(train_samples), len(eval_samples), len(tar_set)))
    for _ in range(10):
        trainIters(in_lang, out_lang, encoder, decoder, train_samples, 1000, max_length, print_every=1000)
        langmod_train2(train_data, langmod, batch_size, bptt, 0, 2000, 0.01)
    trainIters(in_lang, out_lang, encoder, decoder, train_samples, 1000, max_length, print_every=1000)

    encoder.eval()
    decoder.eval()
    langmod.eval()

    corr, tot, acc = evaluateSamples(in_lang, out_lang, encoder, decoder, eval_samples, max_length)
    print('Held-out Accuracy: {0}/{1} = {2}%'.format(corr, tot, 100. * acc))
    return acc


def evalSampleEff(in_lang, out_lang, encoder, decoder, samples, perc, max_length):
    for _ in range(10):
        random.shuffle(samples)

    encoder.train()
    decoder.train()

    train_samples = samples[:int(perc * len(samples))]
    train_forms = set([s[1] for s in train_samples])
    eval_samples = samples[int(perc * len(samples)):]
    eval_forms = set([s[1] for s in eval_samples])
    print('Training with {0}/{1} random data samples'.format(len(train_samples), len(samples)))
    print('{1} Distinct GLTL formulas in training sample: {0}'.format(train_forms, len(train_forms)))
    print('{1} Distinct GLTL formulas in eval sample: {0}'.format(eval_forms, len(eval_forms)))
    trainIters(in_lang, out_lang, encoder, decoder, train_samples, 10000, max_length, print_every=10000)

    encoder.eval()
    decoder.eval()

    corr, tot, acc = evaluateSamples(in_lang, out_lang, encoder, decoder, eval_samples, max_length)
    print('Held-out Accuracy: {0}/{1} = {2}%'.format(corr, tot, 100. * acc))
    return acc
