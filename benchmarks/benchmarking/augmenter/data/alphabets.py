import string



def get_alphabet_from_unicode_ranges(ranges):
    chars = []
    for r in ranges:
        for i in range(r[0], r[1]+1):
            chars.append(chr(i))
    return chars


def chinese_unicode_str():
    # source https://en.wikipedia.org/wiki/CJK_Unified_Ideographs
    ranges = [(0x4E00, 0x9FFF)]  # CJK Unified Ideographs
    return get_alphabet_from_unicode_ranges(ranges)


def japanese_unicode_list():
    # hiragana and katakana only
    ranges = [(0x3041, 0x3096), (0x30A0, 0x30FF)]
    return get_alphabet_from_unicode_ranges(ranges)


def urdu_unicode_list():
    ranges = [
        (0x0600, 0x06FF),
        (0x0750, 0x077F),
        (0xFB50, 0xFDFF),
        (0xFE70, 0xFEFF),
    ]
    return get_alphabet_from_unicode_ranges(ranges)


def korean_unicode_list():
    # source: https://en.wikipedia.org/wiki/Korean_language_and_computers#Hangul_in_Unicode
    ranges = [
        (0xAC00, 0xD7A3),  # Hangul Syllables
        (0x1100, 0x11FF),  # Hangul Jamo
        (0x3130, 0x318F),  # Hangul Compatibility Jamo
        (0xA960, 0xA97F),  # Hangul Jamo Extended-A
        (0xD7B0, 0xD7FF),  # Hangul Jamo Extended-B
    ]
    return get_alphabet_from_unicode_ranges(ranges)


ENGLISH_CHARACTERS = list(string.ascii_lowercase)
SPANISH_CHARACTERS = list("abcdefghijklmnñopqrstuvwxyz")
GERMAN_CHARCTERS = list("abcdefghijklmnopqrstuvwxyzäöüß")
FRENCH_CHARACTERS = list("abcdefghijklmnopqrstuvwxyzàâäéèêëîïôöùûüç")
JAPANESE_CHARACTERS = japanese_unicode_list()
CHINESE_CHARACTERS = chinese_unicode_str()
URDU_CHARACTERS = urdu_unicode_list()
KOREAN_CHARACTERS = korean_unicode_list()