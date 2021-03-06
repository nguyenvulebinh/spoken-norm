import difflib
import regtag
import random


def merge_span(words, tags):
    spans, span_tags = [], []
    current_tag = 'O'
    span = []
    for w, t in zip(words, tags):
        w = w.strip(":-")
        if len(w) == 0:
            continue
        t_info = t.split('-')
        if t_info[-1] != current_tag or t_info[0] == 'B':
            if len(span) > 0:
                spans.append(' '.join(span))
                span_tags.append(current_tag)
            span = [w]
            current_tag = t_info[-1]
        else:
            span.append(w)
    if len(span) > 0:
        spans.append(' '.join(span))
        span_tags.append(current_tag)
    return spans, span_tags


def make_spoken(text, do_split=True):
    src, tgt = [], []
    if do_split:
        chunk_size = random.choice(list(range(0, 10)) + list(range(10, 35)) * 4)
        if chunk_size > 0:
            text = random.choice(split_chunk_input(text, chunk_size))
        else:
            text = ''
    words, word_tags = merge_span(*regtag.tagging(text))
    for span, t in zip(words, word_tags):
        if t == 'O':
            for w in span.split():
                w = w.strip('/.,?!').lower()
                if len(w) > 0:
                    src.append(w)
                    tgt.append(w)
                if random.random() < 0.01:
                    random_value = regtag.augment.get_random_span()
                    tgt.append(random_value[0])
                    src.append(random_value[1].lower())
        else:
            random_value = regtag.augment.get_random_span(t, span.lower())
            tgt.append(random_value[0])
            src.append(random_value[1].lower())

    if len(src) == 0:
        tgt, src = regtag.get_random_span()
        src = [src]
        tgt = [tgt]

    return src, tgt


def split_chunk_input(raw_text, chunk_size):
    input_words = raw_text.strip().split()
    clean_data = [input_words[i:i + chunk_size] for i in range(0, len(input_words), chunk_size)]
    if len(clean_data) > 1:
        clean_data = [" ".join(clean_data[i] + clean_data[i + 1]) for i in range(len(clean_data) - 1)]
    else:
        clean_data = [" ".join(clean_data[0])]
    return clean_data


def equalize(s1, s2):
    l1 = s1.split()
    l2 = s2.split()
    res1 = []
    res2 = []
    combine = []
    prev = difflib.Match(0, 0, 0)
    for match in difflib.SequenceMatcher(a=l1, b=l2).get_matching_blocks():
        if prev.a + prev.size != match.a:
            for i in range(prev.a + prev.size, match.a):
                res2 += ['_' * len(l1[i])]
            res1 += l1[prev.a + prev.size:match.a]

            for i in l1[prev.a + prev.size:match.a]:
                if len(combine) < len(l1) // 2:
                    print(l1[prev.a + prev.size:match.a])
                    combine.append(i)
        if prev.b + prev.size != match.b:
            for i in range(prev.b + prev.size, match.b):
                res1 += ['_' * len(l2[i])]
            res2 += l2[prev.b + prev.size:match.b]

            for i in l2[prev.b + prev.size:match.b]:
                if len(combine) >= len(l2) // 2:
                    print(l2[prev.b + prev.size:match.b])
                    combine.append(i)
        res1 += l1[match.a:match.a + match.size]
        res2 += l2[match.b:match.b + match.size]
        combine += l2[match.b:match.b + match.size]
        prev = match
    return ' '.join(res1), ' '.join(res2), combine


def count_overlap(words_1, words_2):
    # print(words_1, words_2)
    assert len(words_1) == len(words_2)
    len_overlap = 0
    for match in difflib.SequenceMatcher(a=words_1, b=words_2).get_matching_blocks():
        len_overlap += match.size

    # for w1, w2 in zip(words_1, words_2):
    #     if w1 == w2:
    #         len_overlap += 1
    return len_overlap


def find_overlap_chunk(txt_1, txt_2):
    # print(txt_1)
    # print(txt_2)
    window_view = 1
    idx_1 = len(txt_1) - window_view
    idx_2 = window_view
    over_lap = 0
    current_best_idx_1 = len(txt_1)
    current_best_idx_2 = 0

    while window_view <= len(txt_1) and window_view <= len(txt_2):
        current_overlap = count_overlap(txt_1[idx_1:], txt_2[:idx_2])
        print(current_overlap)
        if over_lap < current_overlap:
            over_lap = current_overlap
            current_best_idx_1 = idx_1
            current_best_idx_2 = idx_2
        window_view += 1
        idx_1 = len(txt_1) - window_view
        idx_2 = window_view
        # else:
        #     break
    print('----->', txt_1[current_best_idx_1:], txt_2[:current_best_idx_2])
    return txt_1[current_best_idx_1:], txt_2[:current_best_idx_2]


def concat_chunks(list_chunks):
    concat_string = list_chunks[0].split()
    for i in range(1, len(list_chunks)):
        remain_string = list_chunks[i].split()
        s1, s2 = find_overlap_chunk(concat_string, remain_string)
        s1 = ' '.join(s1)
        s2 = ' '.join(s2)
        _, _, overlap_merged = equalize(s1, s2)
        merge_len = len(s1.split())

        concat_string = concat_string[:len(concat_string) - merge_len] + overlap_merged + remain_string[merge_len:]

    concat_string = ' '.join(concat_string)
    return concat_string


if __name__ == "__main__":
    chunks = [
        'vi???c tr??? l????ng trong gi???i ng??n h??ng ch??nh l???ch m???t tr???i m???t v???c gi???a ng??n h??ng c??? ph???n t?? nh??n v???i c??c ng??n h??ng c??? ph???n nh?? n?????c nh?? n?????c',
        'h??ng c??? ph???n t?? nh??n v???i c??c ng??n h??ng c??? ph???n nh?? n?????c nh?? n?????c n???m c??? ph???n chi ph???i c??ng l?? xi i ??u nh??ng t???i b???n ng??n h??ng',
        'n???m c??? ph???n chi ph???i c??ng l?? xi i ??u nh??ng t???i b???n ng??n h??ng vi???t com banh bi ai ??i vi vi???t tin banh a g??? ri banh th??',
        'vi???t com banh bi ai ??i vi vi???t tin banh a g??? ri banh th?? m???c l????ng c???a c??c xi i ??u v???n ???????c nh?? n?????c ki???m so??t']

    chunks_formated = ['vingroup gia nh???p di???n ????n kinh t??? th??? gi???i vnmedia h??m nay 1/7/2011 t???p ??o??n vingel',
                       'a h??m nay 1/7/2011 t???p ??o??n vingroup v???i hai d??ng th????ng hi???u ch??nh vincom v?? vinpearl ???? ch??nh th???c',
                       'r??p v???i hai d??ng th????ng hi???u ch??nh vincom v?? vinper ???? ch??nh th???c gia nh???p di???n ????n kinh t??? th??? gi???i world economic forum',
                       'gia nh???p di???n ????n kinh t??? th??? gi???i world economic forum wif v?? tr??? th??nh th??nh vi??n hi???p h???i c??c c??ng ty ph??t',
                       'wif v?? tr??? th??nh th??nh vi??n hi???p h???i c??c c??ng ty ph??t tri???n to??n c???u gold b??? grow company ggc ti??u ch??',
                       'tri???n to??n c???u global grow company ggc ti??u ch?? ????? gia nh???p ggcc ????i h???i c??c th??nh vi??n ph???i c?? ti???m n??ng',
                       '????? gia nh???p ggc ????i h???i c??c th??nh vi??n ph???i c?? ti???m n??ng v?? n???i l???c ph??t tri???n kinh doanh t???m c??? qu???c t???']

    concat_result = concat_chunks(chunks_formated)
    print(concat_result)
