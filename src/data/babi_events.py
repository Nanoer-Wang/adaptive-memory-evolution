"""Strict public bAbI qa1 adapter. Labels never enter statement conversion."""
import re
from src.memory.version_ledger import Evidence


def statement(text, story, line, split):
    match = re.fullmatch(r'(\w+) (?:moved|journeyed|travelled|went) (?:back )?to the (\w+)\.', text)
    if not match:
        raise ValueError(f'Unsupported qa1 statement: {text}')
    owner, value = match.groups()
    return Evidence(owner, 'location', '', value, float(line),
                    f'babi-{split}-{story}', line, 0, len(text), text)


def read_key(question):
    match = re.fullmatch(r'Where is (\w+)\?', question.strip())
    if not match:
        raise ValueError(f'Unsupported qa1 question: {question}')
    return (match.group(1), 'location', '')


def stories(path, split):
    result = []; current = []; story = -1
    for raw in path.read_text(encoding='utf-8').splitlines():
        number, text = raw.split(' ', 1); number = int(number)
        if number == 1:
            if current: result.append(current)
            current = []; story += 1
        if '\t' in text:
            question, answer, support = text.split('\t')
            current.append(dict(question=question, answer=answer,
                                support=[int(v) for v in support.split()], line=number))
        else:
            current.append(statement(text, story, number, split))
    if current: result.append(current)
    return result
