import os


CUR_DIR = os.path.realpath(os.path.dirname(__file__))
DATA_DIR = os.path.join(CUR_DIR, 'data')
DATASETS = ['train', 'validation', 'test']
TOPICS = {'1': 'ordinary_life', 2: 'school_life',
          '3': 'culture_education', '4': 'attitude_emotion',
          '5': 'relationship', '6': 'tourism', '7': 'health',
          '8': 'work', '9': 'politics', '10': 'finance'}
ACTS = {'1': 'inform', '2': 'question', '3': 'directive', '4': 'commissive'}
EMOS = {'0': 'no_emotion', '1': 'anger', '2': 'disgust',
        '3': 'fear', '4': 'happiness', '5': 'sadness', '6': 'surprise'}


def _decode_topic(tag):
    return TOPICS.get(tag)


def _decode_act(act):
    return ACTS.get(act)


def _decode_emo(emo):
    return EMOS.get(emo)


def _file_streams(dataset='train'):
    assert dataset in DATASETS, '!!! MUST BE ONE OF: [{}] !!!'.format(' ,'.join(DATASETS))
    file_paths = ['dialogues_{}.txt',
                  'dialogues_act_{}.txt',
                  'dialogues_emotion_{}.txt']
    file_paths = [os.path.join(DATA_DIR, dataset, i.format(dataset))
                  for i in file_paths]
    return [open(i, 'rb').readlines()
            for i in file_paths]


def _to_unicode(string):
    return str(string, 'utf-8')


def _check_short(string):
    if not len(string) > 0:
        return None
    return string


def _parse_utterances(dial, act, emo, conv_id):
    utters = dial.strip('\n').split('__eou__')
    acts = act.strip('\n').split(' ')
    emos = emo.strip('\n').split(' ')

    # conversation = []
    first_speaker = True
    for utter, act, emo in zip(utters, acts, emos):
        speaker = 'person_a'
        if not first_speaker:
            speaker = 'person_b'
        first_speaker = not first_speaker
        utter = _check_short(utter)
        if utter:
            yield (speaker, utter, _decode_act(act), _decode_emo(emo), conv_id)


def convs(dataset='train'):
    dials, acts, emos = _file_streams(dataset)
    for conv_id, (dial, act, emo) in enumerate(zip(dials, acts, emos)):
        dial, act, emo = [_to_unicode(i) for i in (dial, act, emo)]
        for utterance in _parse_utterances(dial, act, emo, conv_id):
            yield utterance


if __name__ == '__main__':
    things = convs()
    for i in things.__next__():
        print(i)
        print('\n')
