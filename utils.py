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
        'việc trả lương trong giới ngân hàng chênh lệch một trời một vực giữa ngân hàng cổ phần tư nhân với các ngân hàng cổ phần nhà nước nhà nước',
        'hàng cổ phần tư nhân với các ngân hàng cổ phần nhà nước nhà nước nắm cổ phần chi phối cùng là xi i âu nhưng tại bốn ngân hàng',
        'nắm cổ phần chi phối cùng là xi i âu nhưng tại bốn ngân hàng việt com banh bi ai đi vi việt tin banh a gờ ri banh thì',
        'việt com banh bi ai đi vi việt tin banh a gờ ri banh thì mức lương của các xi i âu vẫn được nhà nước kiểm soát']

    chunks_formated = ['vingroup gia nhập diễn đàn kinh tế thế giới vnmedia hôm nay 1/7/2011 tập đoàn vingel',
                       'a hôm nay 1/7/2011 tập đoàn vingroup với hai dòng thương hiệu chính vincom và vinpearl đã chính thức',
                       'rúp với hai dòng thương hiệu chính vincom và vinper đã chính thức gia nhập diễn đàn kinh tế thế giới world economic forum',
                       'gia nhập diễn đàn kinh tế thế giới world economic forum wif và trở thành thành viên hiệp hội các công ty phát',
                       'wif và trở thành thành viên hiệp hội các công ty phát triển toàn cầu gold bồ grow company ggc tiêu chí',
                       'triển toàn cầu global grow company ggc tiêu chí để gia nhập ggcc đòi hỏi các thành viên phải có tiềm năng',
                       'để gia nhập ggc đòi hỏi các thành viên phải có tiềm năng và nội lực phát triển kinh doanh tầm cỡ quốc tế']

    concat_result = concat_chunks(chunks_formated)
    print(concat_result)
