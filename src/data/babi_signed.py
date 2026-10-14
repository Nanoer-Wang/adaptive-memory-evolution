"""Strict parser of bAbI qa9/qa10 assertions, negatives and disjunctions."""
import re


def assertion(text):
    match=re.fullmatch(r'(\w+) is either in the (\w+) or the (\w+)\.',text)
    if match:return match[1],'either',(match[2],match[3])
    match=re.fullmatch(r'(\w+) is (?:no longer|not) in the (\w+)\.',text)
    if match:return match[1],'negative',(match[2],)
    match=re.fullmatch(r'(\w+) (?:is in|(?:moved|went|journeyed|travelled) (?:back )?to) the (\w+)\.',text)
    if match:return match[1],'positive',(match[2],)
    raise ValueError('Unsupported signed statement: '+text)


def question(text):
    match=re.fullmatch(r'Is (\w+) in the (\w+)\?',text.strip())
    if not match:raise ValueError('Unsupported signed question: '+text)
    return match[1],match[2]
