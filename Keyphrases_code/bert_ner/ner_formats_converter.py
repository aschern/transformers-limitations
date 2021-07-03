def convert_bio_to_bioul(input_file, output_file, use_position=False):
    with open(input_file, 'r') as fin:
        lines = fin.readlines()
        with open(output_file, 'w') as fout:
            for i in range(len(lines)):
                if i < len(lines) - 1:
                    cur_row, next_row = lines[i], lines[i + 1]
                if i == len(lines) - 1:
                    cur_row, next_row = lines[i], lines[i]
                if len(cur_row.strip()) == 0:
                    fout.write(cur_row)
                else:
                    if not use_position:
                        token, tag = cur_row.strip().split()
                    else:
                        token, tag, pos = cur_row.strip().split()
                    if len(next_row.strip()) > 0:
                        next_token, next_tag = next_row.strip().split()[:2]
                    else:
                        next_token, next_tag = '', ''
                    if tag.startswith('B') and (not next_tag.startswith('I') or i == (len(lines) - 1)):
                        tag = 'U' + tag[1:]
                    elif tag.startswith('I') and (not next_tag.startswith('I') or i == (len(lines) - 1)):
                        tag = 'L' + tag[1:]
                    if not use_position:
                        fout.write(token + '\t' + tag + '\n')
                    else:
                        fout.write(token + '\t' + tag + '\t' + pos + '\n')


def convert_bioul_to_bio(input_file, output_file, use_position=False):
    with open(input_file, 'r') as fin:
        with open(output_file, 'w') as fout:
            for row in fin.readlines():
                if len(row.strip()) == 0:
                    fout.write(row)
                else:
                    el = row.strip().split()
                    if not use_position:
                        token, tag = row.strip().split()
                    else:
                        token, tag, pos = row.strip().split()
                    if tag.startswith('U'):
                        tag = 'B' + tag[1:]
                    elif tag.startswith('E'):
                        tag = 'I' + tag[1:]
                    if not use_position:
                        fout.write(token + '\t' + tag + '\n')
                    else:
                        fout.write(token + '\t' + tag  + '\t' + pos + '\n')
                        