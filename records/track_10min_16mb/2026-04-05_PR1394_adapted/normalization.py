"""
Dataset Normalization Module

Normalizes text documents according to the following rules:
1. URLs: Keep only major domain (e.g., www.cityofalbany.net/calendar -> a url in www.cityofalbany.net)
2. Email addresses: Convert to "an email address"
3. Asterisk content: Remove * from *text* or **text**
4. Repeated punctuations (>=3): Crop to 3 (e.g., !!!! -> !!!)
5. Excessive caps: Convert to title case (e.g., TIMES -> Times)
6. Boilerplate: Remove boilerplate phrases
7. Gibberish consonants: Crop repeated consonants >3 to 3 (e.g., Ohhhhh -> Ohhh, excluding hmmm)
8. Pipe symbol: Ensure whitespace around |
9. Other punctuations: Ensure one whitespace after punctuation

The detection patterns are reused from dataset_quality_analysis_v7.py.
"""

import re
from typing import Tuple, List


# ==============================================================================
# PATTERN DEFINITIONS (from dataset_quality_analysis_v7.py)
# ==============================================================================

URL_PATTERN = re.compile(
    r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[^\s]*|'
    r'www\.[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}[^\s]*',
    re.IGNORECASE
)

EMAIL_PATTERN = re.compile(
    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
)

# Boilerplate pattern (from dataset_quality_analysis_v7.py)
# Extended to handle copyright symbol with various whitespace patterns
BOILERPLATE_PATTERN = re.compile(
    r'\b(cookie policy|privacy policy|terms of use|terms of service|'
    r'copyright\s*[©\(c\)\[c\]]?\s*(?:(?:19|20)\d{2})?[^.]*(?:all rights reserved)|'
    r'all rights reserved|'
    r'©\s*(?:19|20)\d{2}[^.]*all rights reserved)',
    re.IGNORECASE
)

# Excessive caps: 4+ consecutive uppercase, optionally followed by more uppercase words
# The follow-up words only need 2+ uppercase letters (to catch "TO", "ALL")
EXCESSIVE_CAPS_PATTERN = re.compile(
    r'\b[A-Z]{4,}(?:\s+[A-Z]{2,})*\b'
)

# Gibberish consonants: exclude hmmm-like patterns (h, m repeated)
# Matches 4+ identical consonants in a row (e.g., hhhh = h followed by 3+ more h's)
# Uses word boundaries to avoid partial word matches
REPEATED_CONSONANT_PATTERN = re.compile(
    r'\b\w*?([bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ])\1{3,}\w*?\b',
    re.IGNORECASE
)

# HMM-like pattern to exclude (just h and m repeated)
HMM_PATTERN = re.compile(r'^[hHmM]+$', re.IGNORECASE)

# Placeholder for URL/email protection during normalization
URL_PLACEHOLDER_PREFIX = "__URL_PLACEHOLDER_"
EMAIL_PLACEHOLDER_PREFIX = "__EMAIL_PLACEHOLDER_"
placeholder_counter = [0]


# ==============================================================================
# NORMALIZATION FUNCTIONS
# ==============================================================================

def normalize_url(match: re.Match) -> str:
    """
    Normalize URL to keep only major domain.

    Examples:
        www.cityofalbany.net/calendar -> a url in www.cityofalbany.net
        https://example.com/path -> a url in example.com
        www.carillons.org). -> a url in www.carillons.org
    """
    url = match.group(0)

    # Remove trailing punctuation that's not part of URL
    url = url.rstrip('.,;:!?)')

    # Extract domain
    if url.startswith('http://') or url.startswith('https://'):
        # Remove protocol
        domain = url.split('://', 1)[1]
        # Get domain before first path /
        domain = domain.split('/')[0]
    elif url.startswith('www.'):
        domain = url.split('/')[0]
    else:
        # For other cases, just take before first /
        domain = url.split('/')[0]

    # Remove port if present
    domain = domain.split(':')[0]

    return f"a url in {domain}"


def normalize_email(match: re.Match) -> str:
    """
    Replace email address with placeholder.

    Example:
        onniechilders@msn.com -> an email address
    """
    return "an email address"


def remove_asterisks(text: str) -> str:
    """
    Remove asterisks surrounding content.

    Examples:
        *sign a lot* -> sign a lot
        **sign a lot** -> sign a lot
        *single asterisk -> *single asterisk (no closing *, keep as-is)
    """
    # Handle **text** (double asterisks)
    text = re.sub(r'\*\*([^\*]+)\*\*', r'\1', text)
    # Handle *text* (single asterisks)
    text = re.sub(r'\*([^\*]+)\*', r'\1', text)
    return text


def crop_repeated_punctuation(text: str) -> str:
    """
    Crop repeated punctuations (>=3) to exactly 3.

    Examples:
        !!!! -> !!!
        ???? -> ???
        .... -> ...
        !!!!!!! -> !!!
    """
    def replace_repeated(match):
        punct = match.group(1)
        return punct * 3

    # Match 3+ repetitions of the same punctuation and replace with 3
    return re.sub(r'([!?.])\1{2,}', replace_repeated, text)


def normalize_excessive_caps(match: re.Match) -> str:
    """
    Convert excessive capitalization to title case.

    Examples:
        TIMES -> Times
        IMPORTANT NOTICE -> Important Notice
        COPYRIGHT -> Copyright
    """
    text = match.group(0)
    # Title case: first letter uppercase, rest lowercase
    return text.title()


def remove_boilerplate(match: re.Match) -> str:
    """
    Remove boilerplate phrases entirely.

    Examples:
        "Copyright © 2023 All Rights Reserved" -> ""
        "Privacy Policy" -> ""
    """
    return ""


def should_exclude_from_consonant_normalization(word: str) -> bool:
    """
    Check if word should be excluded from consonant normalization.

    Excludes hmmm-like patterns (just h and m repeated).
    """
    # Remove non-letters for checking
    letters_only = re.sub(r'[^a-zA-Z]', '', word)
    # Check if it's only h and m (case insensitive)
    if letters_only and HMM_PATTERN.match(letters_only):
        return True
    return False


def normalize_gibberish_consonants(match: re.Match) -> str:
    """
    Crop repeated identical consonants to 3 repetitions.

    Examples:
        Ohhhhh -> Ohhh
        Ahhhhhh -> Ahhh
        Brrrrr -> Brrr

    But NOT:
        hmmm -> hmmm (excluded)
        Hmmm -> Hmmm (excluded)
        Mmmm -> Mmmm (excluded)
    """
    full_match = match.group(0)
    consonant = match.group(1)

    # Check if this should be excluded (hmmm-like: only h and/or m in the WHOLE word)
    # Only exclude if the entire word is just h's and m's (case insensitive)
    letters_only = re.sub(r'[^a-zA-Z]', '', full_match)
    if HMM_PATTERN.match(letters_only):
        return full_match

    # Build the replacement: prefix + 3 consonants + suffix
    # Find the start of the repeated consonant sequence
    # The pattern ensures we have at least 4 identical consonants in a row
    pos = 0
    while pos < len(full_match):
        # Look for 4+ identical consonants starting at this position
        if pos < len(full_match) and full_match[pos].lower() == consonant.lower():
            # Count consecutive identical consonants
            count = 1
            while pos + count < len(full_match) and full_match[pos + count].lower() == consonant.lower():
                count += 1
            if count >= 4:
                # Found the sequence
                prefix = full_match[:pos]
                suffix = full_match[pos + count:]
                return prefix + consonant * 3 + suffix
            pos += count
        else:
            pos += 1

    return full_match


def normalize_pipe_whitespace(text: str) -> str:
    """
    Ensure whitespace around pipe symbols.

    Examples:
        A|B -> A | B
        A |B -> A | B
        A| B -> A | B
        A | B -> A | B (unchanged)
    """
    # Add space before | if not present
    text = re.sub(r'([^\s])\|', r'\1 |', text)
    # Add space after | if not present
    text = re.sub(r'\|([^\s])', r'| \1', text)
    return text


def normalize_punctuation_whitespace(text: str) -> str:
    """
    Ensure one whitespace between punctuation and following non-punctuation word.
    If more than one whitespace, crop to one.

    Skips:
        - Decimal numbers (e.g., 1.3)
        - Common abbreviations (e.g., e.g., i.e.)

    Examples:
        Hello.World -> Hello. World
        Hello.  World -> Hello. World
        Hello,World -> Hello, World
        "Hello"world -> "Hello" world
        1.3 -> 1.3 (unchanged)
        e.g.something -> e.g. something
    """

    def should_skip_period_at_pos(text: str, pos: int) -> bool:
        """Check if period at position should be skipped."""
        # Check for decimal number (digit.digit)
        if pos > 0 and pos < len(text) - 1:
            if text[pos - 1].isdigit() and text[pos + 1].isdigit():
                return True

        # Check for abbreviation (e.g., i.e.)
        # Look back up to 5 chars to capture patterns like "e.g."
        lookback = text[max(0, pos - 5):pos + 1]
        # Match e.g. or i.e. with optional preceding space/word char
        if re.search(r'(?:^|\s)(?:e\.g|i\.e)\.$', lookback, re.IGNORECASE):
            return True
        # Also match if we're in the middle of e.g. (the first period after 'e')
        if pos > 0 and pos < len(text) - 1:
            if text[pos - 1].lower() == 'e' and text[pos + 1].lower() == 'g':
                # This is the first period in e.g.
                return True
            if text[pos - 1].lower() == 'i' and text[pos + 1].lower() == 'e':
                # This is the first period in i.e.
                return True

        return False

    # Mark positions to skip
    skip_positions = set()
    for i, char in enumerate(text):
        if char == '.' and should_skip_period_at_pos(text, i):
            skip_positions.add(i)

    # Handle quotes specially: only add space after closing quote (quote preceded by word char)
    # Closing double quote followed immediately by word char -> add space
    text = re.sub(r'(?<=[\w.,;:!?])"([^\s])', r'" \1', text)
    # Closing single quote followed immediately by word char -> add space
    text = re.sub(r"(?<=[\w.,;:!?])'([^\s])", r"' \1", text)

    # After other punctuation (excluding quotes), ensure single space before word
    text = re.sub(r'([,;:!?\)\]\}])([^\s.,;:!?"\'\)\]\}])', r'\1 \2', text)

    # Handle periods with skip check - must check against original position
    # Need to track position through transformations
    result = []
    orig_pos = 0
    for i, char in enumerate(text):
        if char == '.':
            # Check if this position should be skipped
            if orig_pos in skip_positions:
                result.append(char)
                orig_pos += 1
                continue
            # Check if we should add space after this period
            # Look ahead to next non-space char
            next_char_idx = i + 1
            while next_char_idx < len(text) and text[next_char_idx] == ' ':
                next_char_idx += 1

            if next_char_idx < len(text):
                next_char = text[next_char_idx]
                # If next char is a word char (letter), add space
                if next_char.isalpha():
                    result.append('. ')
                    orig_pos += 1
                    continue
            result.append(char)
            orig_pos += 1
        else:
            result.append(char)
            orig_pos += 1

    text = ''.join(result)

    # Collapse multiple spaces to single space
    text = re.sub(r' +', ' ', text)

    return text


# ==============================================================================
# MAIN NORMALIZATION PIPELINE
# ==============================================================================

def normalize_document(text: str) -> str:
    """
    Apply all normalization rules to a document.

    Uses placeholder strategy for URLs/emails to protect them from
    punctuation normalization that would break domain names.

    Order of operations:
    1. Replace URLs/emails with placeholders
    2. Remove boilerplate
    3. Remove asterisks
    4. Normalize repeated consonants
    5. Normalize excessive caps
    6. Crop repeated punctuation
    7. Fix pipe whitespace
    8. Fix general punctuation whitespace
    9. Restore URLs/emails from placeholders

    Args:
        text: Input text to normalize

    Returns:
        Normalized text
    """
    placeholders = {}
    placeholder_counter[0] = 0

    def make_placeholder(prefix: str, value: str) -> str:
        idx = placeholder_counter[0]
        placeholder_counter[0] += 1
        ph = f"{prefix}{idx}__"
        placeholders[ph] = value
        return ph

    # Step 1: Replace URLs with placeholders
    def url_replacer(match):
        normalized = normalize_url(match)
        return make_placeholder(URL_PLACEHOLDER_PREFIX, normalized)

    text = URL_PATTERN.sub(url_replacer, text)

    # Step 2: Replace emails with placeholders
    def email_replacer(match):
        normalized = normalize_email(match)
        return make_placeholder(EMAIL_PLACEHOLDER_PREFIX, normalized)

    text = EMAIL_PATTERN.sub(email_replacer, text)

    # Step 3: Remove boilerplate
    text = BOILERPLATE_PATTERN.sub(remove_boilerplate, text)

    # Step 4: Remove asterisks
    text = remove_asterisks(text)

    # Step 5: Normalize gibberish consonants
    text = REPEATED_CONSONANT_PATTERN.sub(normalize_gibberish_consonants, text)

    # Step 6: Normalize excessive caps
    text = EXCESSIVE_CAPS_PATTERN.sub(normalize_excessive_caps, text)

    # Step 7: Crop repeated punctuation
    text = crop_repeated_punctuation(text)

    # Step 8: Normalize pipe whitespace
    text = normalize_pipe_whitespace(text)

    # Step 9: Normalize general punctuation whitespace
    text = normalize_punctuation_whitespace(text)

    # Step 10: Restore URLs and emails from placeholders
    # Restore in reverse order to handle any edge cases
    for ph, value in sorted(placeholders.items(), key=lambda x: -len(x[0])):
        text = text.replace(ph, value)

    # Clean up any double spaces that might have been created
    text = re.sub(r' +', ' ', text)
    text = text.strip()

    return text


# ==============================================================================
# TEST CASES
# ==============================================================================

def run_tests():
    """Run comprehensive test cases for all normalization functions."""

    print("=" * 70)
    print("Running Normalization Tests")
    print("=" * 70)

    test_cases = [
        # URL tests
        {
            'name': 'URL with subdomain and path',
            'input': 'Visit www.cityofalbany.net/calendar for events.',
            'expected': 'Visit a url in www.cityofalbany.net for events.',
            'category': 'URL'
        },
        {
            'name': 'URL with trailing punctuation',
            'input': 'See www.carillons.org). for more info.',
            'expected': 'See a url in www.carillons.org for more info.',
            'category': 'URL'
        },
        {
            'name': 'HTTPS URL',
            'input': 'Go to https://example.com/path/to/page for details.',
            'expected': 'Go to a url in example.com for details.',
            'category': 'URL'
        },
        {
            'name': 'HTTP URL',
            'input': 'Check http://test.org/page.',
            'expected': 'Check a url in test.org',
            'category': 'URL'
        },
        {
            'name': 'Multiple URLs',
            'input': 'Visit www.site1.com/page and www.site2.org/another.',
            'expected': 'Visit a url in www.site1.com and a url in www.site2.org',
            'category': 'URL'
        },

        # Email tests
        {
            'name': 'Simple email',
            'input': 'Contact onniechilders@msn.com for details.',
            'expected': 'Contact an email address for details.',
            'category': 'Email'
        },
        {
            'name': 'Email with dots in local part',
            'input': 'Email john.doe@example.co.uk please.',
            'expected': 'Email an email address please.',
            'category': 'Email'
        },

        # Asterisk tests
        {
            'name': 'Single asterisks',
            'input': '*sign a lot* is the phrase.',
            'expected': 'sign a lot is the phrase.',
            'category': 'Asterisk'
        },
        {
            'name': 'Double asterisks',
            'input': '**sign a lot** is emphasized.',
            'expected': 'sign a lot is emphasized.',
            'category': 'Asterisk'
        },
        {
            'name': 'Multiple asterisk groups',
            'input': '*first* and **second** are here.',
            'expected': 'first and second are here.',
            'category': 'Asterisk'
        },
        {
            'name': 'Unmatched single asterisk',
            'input': '*single asterisk remains',
            'expected': '*single asterisk remains',
            'category': 'Asterisk'
        },

        # Repeated punctuation tests
        {
            'name': 'Four exclamations',
            'input': 'Wow!!!! That is great.',
            'expected': 'Wow!!! That is great.',
            'category': 'Repeated Punctuation'
        },
        {
            'name': 'Many exclamations',
            'input': 'Amazing!!!!!!!',
            'expected': 'Amazing!!!',
            'category': 'Repeated Punctuation'
        },
        {
            'name': 'Four question marks',
            'input': 'What???? Really?',
            'expected': 'What??? Really?',
            'category': 'Repeated Punctuation'
        },
        {
            'name': 'Four periods',
            'input': 'Wait.... Then continue.',
            'expected': 'Wait... Then continue.',
            'category': 'Repeated Punctuation'
        },
        {
            'name': 'Three punctuations unchanged',
            'input': 'Exactly!!!',
            'expected': 'Exactly!!!',
            'category': 'Repeated Punctuation'
        },

        # Excessive caps tests
        {
            'name': 'Simple all caps',
            'input': 'The TIMES newspaper.',
            'expected': 'The Times newspaper.',
            'category': 'Excessive Caps'
        },
        {
            'name': 'Multiple caps words',
            'input': 'IMPORTANT NOTICE TO ALL.',
            'expected': 'Important Notice To All.',
            'category': 'Excessive Caps'
        },
        {
            'name': 'COPYRIGHT caps',
            'input': 'COPYRIGHT 2023 by Author.',
            'expected': 'Copyright 2023 by Author.',
            'category': 'Excessive Caps'
        },

        # Boilerplate tests
        {
            'name': 'Copyright boilerplate',
            'input': 'Article content. Copyright © 2023 All Rights Reserved. More content.',
            'expected': 'Article content. . More content.',
            'category': 'Boilerplate'
        },
        {
            'name': 'Privacy policy',
            'input': 'Please read our Privacy Policy before continuing.',
            'expected': 'Please read our before continuing.',
            'category': 'Boilerplate'
        },
        {
            'name': 'Terms of service',
            'input': 'By using this site you agree to our Terms of Service.',
            'expected': 'By using this site you agree to our .',
            'category': 'Boilerplate'
        },

        # Gibberish consonant tests
        {
            'name': 'Ohhhhh pattern',
            'input': 'Ohhhhh my gosh!',
            'expected': 'Ohhh my gosh!',
            'category': 'Gibberish Consonants'
        },
        {
            'name': 'Ahhhhhh pattern',
            'input': 'Ahhhhhh I see.',
            'expected': 'Ahhh I see.',
            'category': 'Gibberish Consonants'
        },
        {
            'name': 'Brrrrr pattern',
            'input': 'It is cold brrrrr outside.',
            'expected': 'It is cold brrr outside.',
            'category': 'Gibberish Consonants'
        },
        {
            'name': 'Hmmm excluded',
            'input': 'Hmmm let me think.',
            'expected': 'Hmmm let me think.',
            'category': 'Gibberish Consonants'
        },
        {
            'name': 'hmmm excluded lowercase',
            'input': 'hmmm interesting.',
            'expected': 'hmmm interesting.',
            'category': 'Gibberish Consonants'
        },

        # Pipe symbol tests
        {
            'name': 'Pipe without spaces',
            'input': 'A|B|C',
            'expected': 'A | B | C',
            'category': 'Pipe Whitespace'
        },
        {
            'name': 'Pipe with space only after',
            'input': 'A| B',
            'expected': 'A | B',
            'category': 'Pipe Whitespace'
        },
        {
            'name': 'Pipe with space only before',
            'input': 'A |B',
            'expected': 'A | B',
            'category': 'Pipe Whitespace'
        },
        {
            'name': 'Pipe with correct spacing',
            'input': 'A | B',
            'expected': 'A | B',
            'category': 'Pipe Whitespace'
        },

        # Punctuation whitespace tests
        {
            'name': 'Period without space',
            'input': 'Hello.World',
            'expected': 'Hello. World',
            'category': 'Punctuation Whitespace'
        },
        {
            'name': 'Comma without space',
            'input': 'Hello,world',
            'expected': 'Hello, world',
            'category': 'Punctuation Whitespace'
        },
        {
            'name': 'Multiple spaces after punctuation',
            'input': 'Hello.  World',
            'expected': 'Hello. World',
            'category': 'Punctuation Whitespace'
        },
        {
            'name': 'URL domain normalized then punctuated',
            'input': 'Visit www.gmail.com for email.',
            'expected': 'Visit a url in www.gmail.com for email.',
            'category': 'Punctuation Whitespace'
        },
        {
            'name': 'Decimal number unchanged',
            'input': 'The value is 1.3 units.',
            'expected': 'The value is 1.3 units.',
            'category': 'Punctuation Whitespace'
        },
        {
            'name': 'e.g. abbreviation unchanged',
            'input': 'Use e.g. for examples.',
            'expected': 'Use e.g. for examples.',
            'category': 'Punctuation Whitespace'
        },
        {
            'name': 'i.e. abbreviation unchanged',
            'input': 'Use i.e. for clarification.',
            'expected': 'Use i.e. for clarification.',
            'category': 'Punctuation Whitespace'
        },
        {
            'name': 'Quote without space',
            'input': '"Hello"world',
            'expected': '"Hello" world',
            'category': 'Punctuation Whitespace'
        },

        # Combined tests
        {
            'name': 'URL and caps combined',
            'input': 'Visit WWW.EXAMPLE.COM/page for DETAILS.',
            'expected': 'Visit a url in WWW.EXAMPLE.COM for Details.',
            'category': 'Combined'
        },
        {
            'name': 'Complex mixed content',
            'input': 'Contact john@example.com or visit www.site.com/path!!! It is *VERY IMPORTANT*.',
            'expected': 'Contact an email address or visit a url in www.site.com It is Very Important.',
            'category': 'Combined'
        },
        {
            'name': 'Multiple issues in one',
            'input': 'See www.test.org/page... It is *AMAZING*!!!! Contact me@mail.com.',
            'expected': 'See a url in www.test.org It is Amazing!!! Contact an email address.',
            'category': 'Combined'
        },
    ]

    passed = 0
    failed = 0
    categories = {}

    for test in test_cases:
        result = normalize_document(test['input'])
        success = result == test['expected']

        # Track by category
        cat = test['category']
        if cat not in categories:
            categories[cat] = {'passed': 0, 'failed': 0}

        if success:
            passed += 1
            categories[cat]['passed'] += 1
            print(f"✓ {test['name']}")
        else:
            failed += 1
            categories[cat]['failed'] += 1
            print(f"✗ {test['name']}")
            print(f"  Input:    {repr(test['input'])}")
            print(f"  Expected: {repr(test['expected'])}")
            print(f"  Got:      {repr(result)}")

    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    print(f"Total: {passed + failed}, Passed: {passed}, Failed: {failed}")
    print(f"Success Rate: {100 * passed / (passed + failed):.1f}%")

    print("\nBy Category:")
    for cat, stats in sorted(categories.items()):
        total = stats['passed'] + stats['failed']
        print(f"  {cat}: {stats['passed']}/{total} passed")

    print("=" * 70)

    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
