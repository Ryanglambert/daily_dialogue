import numpy as np
import os
import pandas as pd
import textblob


CUR_DIR = os.path.realpath(os.path.dirname(__file__))
DATA_DIR = os.path.join(CUR_DIR, 'data')
DFPATH = os.path.join(DATA_DIR, '{}_data_frame.pkl')

TOPICS = {'1': 'ordinary_life', '2': 'school_life',
          '3': 'culture_education', '4': 'attitude_emotion',
          '5': 'relationship', '6': 'tourism', '7': 'health',
          '8': 'work', '9': 'politics', '10': 'finance'}
ACTS = {'1': 'inform', '2': 'question', '3': 'directive', '4': 'commissive'}
EMOS = {'0': 'no_emotion', '1': 'anger', '2': 'disgust',
        '3': 'fear', '4': 'happiness', '5': 'sadness', '6': 'surprise'}


def _topic_stream():
    topics_file = os.path.join(DATA_DIR, 'dialogues_topic.txt')
    return open(topics_file, 'rb').readlines()


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


def _file_streams():
    file_paths = ['dialogues_text.txt',
                  'dialogues_act.txt',
                  'dialogues_emotion.txt']
    file_paths = [os.path.join(DATA_DIR, i)
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


def _get_convs():
    dials, acts, emos = _file_streams()
    topics = _topic_stream()
    for conv_id, (dial, act, emo, topic) in enumerate(zip(dials, acts, emos, topics)):
        dial, act, emo, topic = [_to_unicode(i) for i in (dial, act, emo, topic)]
        topic = _decode_topic(topic)
        for utterance in _parse_utterances(dial, act, emo):
            yield utterance + (conv_id, topic)


def get_biggest_drawdown(s):
    "returns the largest net decrease in sentiment"
    i = np.argmax(np.maximum.accumulate(s) - s)
    j = np.argmax(s[:i])
    return s[i] - s[j]


def get_biggest_drawup(s):
    "returns the largest net increase in sentiment"
    i = np.argmin(np.minimum.accumulate(s) - s)
    j = np.argmin(s[:i])
    return s[i] - s[j]


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

    df['change_in_polarity'] = df.groupby([df.index.get_level_values(0), 'person'])['polarity'].diff()
    # heard utterance information
    shift_list = ['utter', 'polarity', 'subjectivity', 'change_in_polarity', 'act', 'emo']
    for label in shift_list:
        df['heard_' + label] = df.groupby([df.index.get_level_values(0)])[label].shift()

    df = pd.concat([df, pd.get_dummies(df[['heard_act', 'heard_emo', 'topic']])], axis=1)

    # add biggest drawup and drawdown (largest change in a conversation)
    df['biggest_drawup'] = df.groupby([df.index.get_level_values(0), 'person'])['polarity']\
                             .transform(get_biggest_drawup)
    df['biggest_drawdown'] = df.groupby([df.index.get_level_values(0), 'person'])['polarity']\
                               .transform(get_biggest_drawdown)

    return df


def _data():
    return _make_df(_get_convs())


def _make_train_test_split(df, test_size=1000):
    conv_ids = df.index.get_level_values(0).unique()
    test_indices = np.random.choice(conv_ids, size=test_size, replace=False)
    train_indices = np.setdiff1d(conv_ids, test_indices)
    return df.loc[train_indices], df.loc[test_indices]


def _make_pickles(df, dataset):
    df.to_pickle(DFPATH.format(dataset))


def _read_pickle(dataset):
    return pd.read_pickle(DFPATH.format(dataset))


def get_data(test_size=1000, use_cached=False):
    if use_cached:
        print('Using Cached')
        return _read_pickle('train'), _read_pickle('test')
    else:
        print('Not Using Cached')
        data = _data()
        train, test = _make_train_test_split(data)
        _make_pickles(train, 'train')
        _make_pickles(test, 'test')
    return train, test

