"""Conservative English extraction adapter for a typed-memory development pilot.

Closed schema; unsupported statements are left unparsed, never guessed. Exact
provenance grounding is checked, but does not establish semantic truth/precision.
The generic ledger accepts Evidence from a stronger audited extractor later.
"""
import re
from src.memory.version_ledger import Evidence

VALUE = r'(?P<value>[^.!?,;\n]{2,70}?)'
END = r'(?=[.!?,;\n]|\s+(?:and|but|because|which|where|with|since|for|as)\b|$)'
RULES = [
    ('residence','', 'current', re.compile(r'\bI (?:currently |now )?(?:live|reside) in '+VALUE+END,re.I)),
    ('residence','', 'current', re.compile(r'\bI (?:just |recently )?moved to '+VALUE+END,re.I)),
    ('residence','', 'past', re.compile(r'\bI used to live in '+VALUE+END,re.I)),
    ('residence','', 'planned', re.compile(r'\bI (?:plan|am planning|want) to move to '+VALUE+END,re.I)),
    ('residence','', 'retracted', re.compile(r'\bI (?:no longer live|do not live|don\x27t live) in '+VALUE+END,re.I)),
    ('employer','', 'current', re.compile(r'\bI (?:currently |now )?work (?:at|for) '+VALUE+END,re.I)),
    ('employer','', 'retracted', re.compile(r'\bI no longer work (?:at|for) '+VALUE+END,re.I)),
    ('occupation','', 'current', re.compile(r'\bI (?:currently |now )?work as (?:an? )?'+VALUE+END,re.I)),
    ('age','years','current',re.compile(r'\b(?:I am|I\x27m) (?P<value>\d{1,3}) years old\b',re.I)),
    ('followers','platform','current',re.compile(r'\bI (?:now |currently )?have (?P<hedge>about |around |roughly |over )?(?P<value>\d[\d,]*) followers on (?P<scope>Instagram|Twitter|TikTok|YouTube|Facebook)\b',re.I)),
    ('followers','platform','current',re.compile(r'\bmy (?P<scope>Instagram|Twitter|TikTok|YouTube|Facebook) follower count is (?P<hedge>about |around |roughly |over )?(?P<value>\d[\d,]*)\b',re.I)),
    ('gym_time','routine','current',re.compile(r'\bI (?:usually |normally )?(?:go|head) to the gym[^.!?\n]{0,85}?\bat (?P<value>\d{1,2}(?::\d{2})?\s*[ap]\.?m\.?)',re.I)),
    ('favorite','category','current',re.compile(r'\bmy favorite (?P<scope>color|food|restaurant|movie|book|sport|team) is '+VALUE+END,re.I)),
]
UNTRUSTED_CONTEXT = re.compile(r'\b(?:rephrase|rewrite|translate|pretend|roleplay|act as|compose|hypothetical|fictional|draft|write (?:a|an|me)|example (?:email|letter)|he said|she said|they said)\b',re.I)


def normalize_value(predicate, value):
    value=' '.join(value.strip().split())
    if predicate in ('followers','age'):
        return str(int(value.replace(',','')))
    if predicate=='gym_time':
        m=re.fullmatch(r'(\d{1,2})(?::(\d{2}))?\s*([ap])\.?m\.?',value,re.I)
        if m:
            hour,minute=int(m[1]),int(m[2] or 0)
            if not 1<=hour<=12 or not 0<=minute<60: return None
            return f'{(hour%12)+(12 if m[3].lower()=="p" else 0):02}:{minute:02}'
    return value


def extract_evidence(turns, source_id, observed_at, owner='user'):
    result=[]
    for index,turn in enumerate(turns):
        # Annotation fields are deliberately not read.
        if turn.get('role')!='user': continue
        content=str(turn.get('content',''))
        if UNTRUSTED_CONTEXT.search(content) or '"' in content or '“' in content or '```' in content:
            continue
        for predicate,scope,mode,pattern in RULES:
            for match in pattern.finditer(content):
                value=normalize_value(predicate,match['value'])
                if not value: continue
                actual_scope=match.groupdict().get('scope') or scope
                actual_mode='tentative' if mode=='current' and match.groupdict().get('hedge') else mode
                # A question/hypothetical context is not a personal assertion.
                line_start=max(content.rfind('.',0,match.start()),content.rfind('?',0,match.start()),content.rfind('\n',0,match.start()))+1
                prefix=content[line_start:match.start()].strip()
                if re.search(r'\b(?:if|suppose|imagine|think|maybe|perhaps)\b',prefix,re.I):
                    actual_mode='tentative'
                end=content.find('?',match.end())
                if end>=0 and not re.search(r'[.!\n]',content[match.end():end]):
                    continue
                quote=content[match.start():match.end()]
                result.append(Evidence(owner,predicate,actual_scope.casefold(),value,observed_at,source_id,index,match.start(),match.end(),quote,actual_mode))
    # Within a source conversation preserve utterance/span order, not rule order.
    return sorted(result,key=lambda e:(e.turn,e.start,e.end,e.predicate))


def query_key(question, owner='user'):
    """Read-time routing only; ambiguous/unsupported questions return None."""
    q=question.casefold()
    if not re.search(r'\b(?:i|my|me)\b',q): return None
    if re.search(r'\b(?:friend|wife|husband|mother|father|sister|brother|colleague|son|daughter)\b',q): return None
    # Historical/event-time questions need explicit temporal parsing, not a
    # silent latest-value answer. The pilot abstains from routing these.
    if re.search(r'\b(?:before|previous|used to|last year|last month|in 20\d\d|ago|first|when did)\b',q): return None
    if 'followers' in q or 'follower count' in q:
        platforms=[p for p in ('instagram','twitter','tiktok','youtube','facebook') if p in q]
        return (owner,'followers',platforms[0]) if len(platforms)==1 else None
    if 'gym' in q and any(t in q for t in ('time','when','hour')): return owner,'gym_time','routine'
    if re.search(r'\bwhere\b',q) and re.search(r'\b(?:live|living|reside)\b',q): return owner,'residence',''
    if ('company' in q or 'employer' in q) and 'work' in q: return owner,'employer',''
    if 'how old am i' in q: return owner,'age','years'
    if 'favorite' in q:
        scopes=[s for s in ('color','food','restaurant','movie','book','sport','team') if re.search(r'\b'+s+r'\b',q)]
        if len(scopes)==1: return owner,'favorite',scopes[0]
    return None
