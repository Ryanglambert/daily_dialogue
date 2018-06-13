import os


CUR_DIR = os.path.realpath(os.path.dirname(__file__))
DATA_DIR = os.path.join(CUR_DIR, 'data')
DATASETS = ['train', 'validation', 'test']


def _file_streams(dataset='train'):
    assert dataset in DATASETS, '!!! MUST BE ONE OF: [{}] !!!'.format(' ,'.join(DATASETS))
    file_paths = ['dialogues_{}.txt',
                  'dialogues_act_{}.txt',
                  'dialogues_emotion_{}.txt']
    data_paths = [os.path.join(DATA_DIR, dataset, i.format(dataset)) for i in file_paths]
    return [open(i, 'rb').readlines() for i in data_paths]


def _to_unicode(string):
    return str(string, 'utf-8')


def _marshal_utterance(utter):
    dial, act, emo = utter
    return {'dial': dial,
            'act': act,
            'emo': emo}


def _parse_utterances(dial, act, emo):
    utters = dial.split('__eou__')
    acts = act.strip('\n').split(' ')
    emos = emo.strip('\n').split(' ')

    conversation = []
    first_speaker = True
    for utter, act, emo in zip(utters, acts, emos):
        speaker = 'person_a'
        if not first_speaker:
            speaker = 'person_b'
        first_speaker = not first_speaker
        conversation.append((
            speaker, utter, act, emo
        ))

    return conversation


def convs(dataset='train'):
    dials, acts, emos = _file_streams(dataset)
    for dial, act, emo in zip(dials, acts, emos):
        dial, act, emo = [_to_unicode(i) for i in (dial, act, emo)]
        yield _parse_utterances(dial, act, emo)


things = convs()
for i in things.__next__():
    print(i)
    print('\n')