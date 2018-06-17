import os
import pandas as pd
import textblob


CUR_DIR = os.path.realpath(os.path.dirname(__file__))
DATA_DIR = os.path.join(CUR_DIR, 'data')
DATASETS = ['train', 'validation', 'test']
TOPICS_FILE = os.path.join(DATA_DIR, 'dialogues_topic.txt')

TOPICS = {'1': 'ordinary_life', '2': 'school_life',
          '3': 'culture_education', '4': 'attitude_emotion',
          '5': 'relationship', '6': 'tourism', '7': 'health',
          '8': 'work', '9': 'politics', '10': 'finance'}
ACTS = {'1': 'inform', '2': 'question', '3': 'directive', '4': 'commissive'}
EMOS = {'0': 'no_emotion', '1': 'anger', '2': 'disgust',
        '3': 'fear', '4': 'happiness', '5': 'sadness', '6': 'surprise'}


def _topic_stream(dataset='train'):
    assert dataset in DATASETS, '!!! MUST BE ONE OF: [{}] !!!'.format(' ,'.join(DATASETS))
    key = {'train': (0, 11118),
           'validation': (11118, 12118),
           'test': (12118, 13118)}
    subset = key[dataset]
    start, end = subset
    stream = open(TOPICS_FILE, 'rb').readlines()
    return stream[start: end]


def _decode_topic(tag):
    tag = tag.strip('\n')
    stuff = TOPICS.get(tag)
    if not stuff:
        raise Exception(tag)
    else:
        return stuff


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


def _sentiment(string):
    blob = textblob.TextBlob(string)
    polarity, subjectivity = blob.sentiment
    return polarity, subjectivity


def _parse_utterances(dial, act, emo):
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
            polarity, subjectivity = _sentiment(utter)
            yield (speaker, utter, _decode_act(act), _decode_emo(emo), polarity, subjectivity)


def _get_convs(dataset='train'):
    dials, acts, emos = _file_streams(dataset)
    topics = _topic_stream(dataset)
    for conv_id, (dial, act, emo, topic) in enumerate(zip(dials, acts, emos, topics)):
        dial, act, emo, topic = [_to_unicode(i) for i in (dial, act, emo, topic)]
        topic = _decode_topic(topic)
        for utterance in _parse_utterances(dial, act, emo):
            yield utterance + (conv_id, topic)


def _make_df(convs):
    df = pd.DataFrame(convs,
                      columns=['person',
                               'utter',
                               'act',
                               'emo',
                               'polarity',
                               'subjectivity',
                               'conv',
                               'topic'])
    df.set_index(['conv'], inplace=True)
    df.set_index([df.index, df.groupby(df.index).cumcount()], inplace=True)
    return df


def _data(dataset):
    return _make_df(_get_convs(dataset))


def train():
    return _data('train')


def validation():
    return _data('validation')


def test():
    return _data('test')


if __name__ == '__main__':
    things = _get_convs()
    for i in things.__next__():
        print(i)
        print('\n')
