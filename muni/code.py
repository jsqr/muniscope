from enum import Enum
from dataclasses import dataclass, field
import re

###################
# Because the outline structure of most legal codes is so simple, it's feasible
# to implement a parser using a hand-coded state machine that shifts between
# states according to the level of the outline. This may be a simpler approach
# than specifying a BNF-style grammar for Lark or a similar parser generator,
# because in some cases outlines skip levels, which would complicate the formal
# grammar.

class Level(Enum):
    H0 = 0 # top level (initial state)
    H1 = 1
    H2 = 2
    H3 = 3

@dataclass
class HeadingPattern:
    level: Level
    regex: str
    multi_line: bool # whether the heading spans multiple lines

@dataclass
class Heading:
    level: Level
    # heading_type: str # e.g. "section", "subsection", "article", "chapter"
    enumeration: str # number or letter (e.g. "1", "a", "i", "A", "XVII")
    heading_text: str

@dataclass
class Segment:
    level: Level
    headings: dict[Level, Heading|None] = field(default_factory=dict)
    paragraphs: list[str] = field(default_factory=list) # list of paragraphs
    chunks: list[str] = field(default_factory=list)

    def chunkify(self, n: int):
        """Regroup the paragraphs of a segment into chunks of size no more than n characters."""
        self.chunks = []
        current_chunk = ''
        for paragraph in self.paragraphs:
            if len(current_chunk) + len(paragraph) > n:
                self.chunks.append(current_chunk)
                current_chunk = ''
            current_chunk += paragraph + '\n'
        self.chunks.append(current_chunk)

## For our purposes, a document is just a list of segments -- the structure is
## implicit in the headings, which will be uploaded to a relational database

def split_paragraph(paragraph: str) -> tuple[str, str]:
    """"Split a paragraph into its first line and the rest of the paragraph."""
    lines = paragraph.split('\n', 1)
    if len(lines) == 0:
        return '', ''
    first_line = lines[0]
    rest_of_paragraph = lines[1] if len(lines) > 1 else ''
    return first_line, rest_of_paragraph

def match_heading(paragraph: str, patterns: dict[Level, HeadingPattern]) -> Heading | None:
    """For each patern in `patterns`, check if the paragraph matches (e.g., pattern '^Chapter [IVXLC]+'
    matches 'Chapter VII'). If a match is found, return a Heading object. Otherwise, return None."""
    
    paragraph = paragraph.strip()

    for level, pattern in patterns.items():
        pattern_regex = re.compile(pattern.regex, re.DOTALL)
        if pattern.multi_line:
            first, rest = split_paragraph(paragraph)
            match = pattern_regex.match(first)
            if match:
                return Heading(level=level, enumeration=match.group(1), heading_text=rest)
        else:
            match = pattern_regex.match(paragraph)
            if match:
                return Heading(level=level, enumeration=match.group(1), heading_text=match.group(2))

class StateMachineParser:
    def __init__(self, document_name: str, heading_patterns: dict[Level, HeadingPattern]):
        self.document = []
        self.heading_names = {Level.H0: document_name, Level.H1: None, Level.H2: None, Level.H3: None}
        self.patterns = heading_patterns
        self.state = Level.H0

    def parse(self, text):
        paragraphs = text.split('\n\n')

        segment_headings = {Level.H0: self.heading_names[Level.H0]}
        segment = Segment(level=Level.H0, headings=segment_headings, paragraphs=[]) # preamble

        for paragraph in paragraphs:

            match = match_heading(paragraph, self.patterns)

            # no heading found, so add paragraph to the current segment
            if not match:
                segment.paragraphs.append(paragraph)
                continue

            # found a heading!
            self.document.append(segment) # add the last segment to document

            self.state = match.level
            new_headings = segment.headings.copy()
            new_headings[match.level] = match
            for level in Level: # have to delete the headings at higher levels in case of skips later
                if level.value in new_headings and level.value > match.level.value:
                    del new_headings[level]
            segment = Segment(level=self.state, headings=new_headings, paragraphs=[]) # start a new segment
        return self.document
    
    def summarize_matches(self, text):
        paragraphs = text.split('\n\n')
        for paragraph in paragraphs:
            match = match_heading(paragraph, self.patterns)
            if match:
                print(f"{' ' * 2 * match.level.value}{match.level.name} heading: {match.enumeration} {match.heading_text}")

##################################################
## FIXME: consider moving to a separate llm  tools module?

import marvin
from openai import OpenAI

openai_client = OpenAI()

def llm(prompt: str, system: str = ""):
    chat_completion = openai_client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": system
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        model="gpt-4o",
    )
    return chat_completion.choices[0].message.content

@marvin.fn
def infer_regex(examples: list[str]) -> str:
    """
    Return a regular expression matching the provided examples, which are the first
    lines of headings in a document. Return a string representing the regular expression,
    according to the following Guidelines and Examples
    
    **Guidelines:**

    1. The regular expression should match the examples and any similar headings,
    and should not match unrelated text.
    2. Terms indicating the type of the heading, such as "Chapter", "Section", "Article", "Title", etc.,
    should be included verbatim in the regular expression.
    3. Assume that enumeration numbers and letters in the pattern can take on a normal range
    (e.g., if you see 1, 2, 3 as examples you should allow other digits like 7 in the match).
    4. Enumerations should be captured by a group in the regular expression. Enumerations may be
    expressions with numbers, letters, or compounds suchs as '5-2'. Text following the enumeration
    should also be captured by a group.
    5. Assume that capitalization is consistent within the document.
    6. The regular expression should be PCRE-compatible.
    7. The regular expression should match the beginning of the line with '^',
    and the end of a line with '$'.
    8. There should be no newlines in the regular expressions (the examples will just be single lines).

    **Examples:**

    1. infer_regex(["Chapter 1: Introduction", "Chapter 2: The Basics"]) -> '^Chapter (\\d+): (.+)$'
    2. infer_regex(["Article I: Scope", "Article II: Definitions"]) -> '^Article ([IVXLC]+): (.+)$'
    3. infer_regex(["Title A: General Provisions", "Title B: Administration"]) -> '^Title ([A-Z]+): (.+)$'
    4. infer_regex(["Section 1-1: Purpose", "Section 1-2: Definitions"]) -> '^Section (\\d+-\\d+): (.+)$'
    5. infer_regex(["4-1: Purpose", "4-2: Definitions"]) -> '^(\\d+-\\d+): (.+)$'

    Your regular expresion (just the expression string itself, without quotation marks or any other text):
    """

def infer_regex_stripped(examples: list[str]) -> str:
    first_lines = [example.split('\n')[0] for example in examples]
    inferred = infer_regex(first_lines)
    return inferred.strip("\'\"") # needed because the LLM ignores instructions

def is_multi_line(examples: list[str]) -> bool:
    return any('\n' in example.strip() for example in examples)

def infer_heading_patterns(example_headings: dict[Level, list[str]]) -> dict[Level, HeadingPattern]:
    """Infer heading patterns from examples. Return a dictionary mapping levels to
    HeadingPattern objects."""
    return {k: HeadingPattern(level=k, regex=infer_regex_stripped(v), multi_line=is_multi_line(v))
            for k, v in example_headings.items()}

@marvin.fn
def infer_level_name(pattern: HeadingPattern) -> str:
    """Infer level names from the regular expressions in the patterns. For example,
    if pattern.regex is '^Title \\d+$', the level name would be 'Title'. The name
    should be a string starting with a capital letter, followed by lowercase letters,
    with no punctuation. If there is no clear name (e.g., if the pattern is
    '^\\d+\\-\\d+\\-\\d+'), return 'Section'.
    """

# shouldn't be necessary, but the LLM sometimes ignores the instructions about letters
def letters_only(s: str) -> str:
    return ''.join(c for c in s if c.isalpha())

def infer_level_names(patterns: dict[Level, HeadingPattern]) -> dict[Level, str]:
    return {k: letters_only(infer_level_name(v)) for k, v in patterns.items()}

@dataclass
class Jurisdiction:
    name: str
    patterns: dict[Level, HeadingPattern]
    source_local: str = ''
    source_url: str = ''
    raw_text: str = ''
    parser: StateMachineParser | None = None
    document: list[Segment] = field(default_factory=list)
    autoload: bool = True    

    def __post_init__(self):
        if self.autoload:
            self.load()

    def load(self):
        """Loads the text of local code from file (source_local)."""
        try:
            with open(self.source_local, "r") as f:
                self.raw_text = f.read()
        except FileNotFoundError as e:
            print(f"Error reading {self.source_local}: {e}")

def summarize_document(document: list[Segment]):
    for segment in document:
        if len(segment.paragraphs) == 0:
            continue
        text = '\n'.join(segment.paragraphs)
        print(f"{segment.level}: {len(segment.paragraphs)} paragraphs, {len(text)} characters")

def chunkify_document(document: list[Segment], n: int):
    for segment in document:
        if len(segment.paragraphs) == 0:
            continue
        segment.chunkify(n)

def summarize_chunks(document: list[Segment]):
    for segment in document:
        for i, chunk in enumerate(segment.chunks):
            print(f"{segment.level} chunk {i}: {len(chunk)} characters")
