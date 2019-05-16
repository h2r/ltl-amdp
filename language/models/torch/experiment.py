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
#print(random.choice(pairs))

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


def text_to_ltl(lang, ltl):
    # add path to the checkpoint folder below, this will load the encoder and decoder models
    path = '../../models/torch/checkpoint/'
    #path = '../../models/torch/checkpoint-gpu/'

    print('Loading checkpoints')

    print(encoder1.state_dict)
    print(attn_decoder1.state_dict)
    encoder1.load_state_dict(torch.load(path + 'encoder'))
    attn_decoder1.load_state_dict(torch.load(path + 'decoder'))
    print('Finished loaded checkpoints!')
    # set it to evaluate mode
    encoder1.eval()
    attn_decoder1.eval()

    # evaluate sentences in 'pairs', where pairs is a list of all the commands [ [natural language, ltl] ]
    # e.g., pairs = [['proceed to the green room by going through the blue room .', 'F ( blue_room & F ( red_room ) )']]

    pairs = [[lang, ltl]]
    print('Pairs, ', pairs)
    ltl = evaluateSelected(input_lang, output_lang, encoder1, attn_decoder1, pairs, MAX_LENGTH)

    return ltl

if __name__ == '__main__':

    lang = 'proceed to the green room by going through the blue room .'
    true_ltl = 'F ( green_room & F ( blue_room ) )'

    #lang = 'go to the blue room .'
    #true_ltl = 'F ( blue_room )'

    ltl = text_to_ltl(lang, true_ltl)
    print('Model output of LTL, ', ltl)
