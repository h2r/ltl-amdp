import math
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


#
# def trainLangmod(input_variable, target_variable, langmod, langmod_optimizer, criterion):
#     langmod_hidden = langmod.initHidden()
#     langmod_optimizer.zero_grad()
#
#     loss = 0.0
#
#     langmod_output, langmod_hidden = langmod(input_variable, langmod_hidden)
#     topv, topi = langmod_output.data.topk(1)
#     loss += criterion(langmod_output, target_variable[0])
#
#     loss.backward()
#
#     langmod_optimizer.step()
#
#     return loss.data[0]
#
#
# def trainLangmodIters(langmod, lang, samples, n_iters, print_every=100, learning_rate=0.001):
#     start = time.time()
#     print_loss_total = 0
#
#     langmod_optimizer = optim.Adam(langmod.parameters(), lr=learning_rate)
#     train_data = itertools.cycle(iter([[SOS_token] + indexesFromSentence(lang, s) + [EOS_token] for s in samples]))
#     criterion = nn.NLLLoss()
#     for i in range(1, n_iters + 1):
#         indexes = train_data.next()
#         for j in range(1, len(indexes)):
#             input_variable = Variable(torch.LongTensor([indexes[j - 1]]).view(-1, 1))
#             if use_cuda:
#                 input_variable = input_variable.cuda()
#             target_variable = Variable(torch.LongTensor([indexes[j]]).view(-1, 1))
#             if use_cuda:
#                 target_variable = target_variable.cuda()
#
#             loss = trainLangmod(input_variable, target_variable, langmod, langmod_optimizer, criterion)
#             print_loss_total += loss
#
#         if i % print_every == 0:
#             print_loss_avg = print_loss_total / print_every
#             print_loss_total = 0
#             print('%s (%d %d%%) %.4f' % (timeSince(start, i / n_iters), i, i / n_iters * 100, print_loss_avg))


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.show()
