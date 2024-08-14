import re


def extract_string_from_dict(string: str, key: str) -> str:
    matches = re.findall(r'\{?' + key + r'\s*[:=]\s*(.+?)(?=\s*}|$)', string)
    if len(matches) == 0:
        return None
    else:
        ret = re.sub(r'\s+', ' ', matches[0])
        ret = re.sub(r'[^a-zA-Z0-9 ]', '', ret)
    return ret

def string_match(source: str, target: str) -> bool:
    source = source.lower()
    target = target.lower()
    source = re.sub(r'\s+', ' ', source)
    target = re.sub(r'\s+', ' ', target)
    source = re.sub(r'[^a-zA-Z0-9]', '', source)
    target = re.sub(r'[^a-zA-Z0-9]', '', target)
    if target is None or target == '':
        return False
    if target == source or target in source or source in target:
        return True
    else:
        return False