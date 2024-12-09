from enum import Enum, unique
from dataclasses import dataclass, field
import re
from warnings import warn

###################
# Because the outline structure of most legal codes is so simple, it's feasible
# to implement a parser using a hand-coded state machine that shifts between
# states according to the level of the outline. This may be a simpler approach
# than specifying a BNF-style grammar for Lark or a similar parser generator,
# because in some cases outlines skip levels, which would complicate the formal
# grammar.

#@unique
#class Level(Enum):
#    H0 = 0 # top level (initial state)
#    H1 = 1
#    H2 = 2
#    H3 = 3
#    H4 = 4
#    H5 = 5

LEVELS = [0, 1, 2, 3, 4, 5]

@dataclass
class HeadingPattern:
    level: int #Level
    regex: str
    multi_line: bool # whether the heading spans multiple lines

@dataclass
class Heading:
    level: int #Level
    # level_name: str # e.g. "Title", "Chapter", "Article", "Section"
    enumeration: str # number or letter (e.g. "1", "a", "i", "A", "XVII")
    heading_text: str

@dataclass
class Segment:
    level: int #Level
    headings: dict[int, Heading] = field(default_factory=dict)
    paragraphs: list[str] = field(default_factory=list) # list of paragraphs
    chunks: list[str] = field(default_factory=list) # sized for embeddings

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

def match_heading(paragraph: str, patterns: dict[int, HeadingPattern]) -> Heading | None:
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
    def __init__(self, document_name: str, heading_patterns: dict[int, HeadingPattern]):
        self.document = []
        self.document_name = document_name # heading_names = {Level.H0: document_name}
        self.patterns = heading_patterns
        self.state = 0 #Level.H0

    def parse(self, text):
        paragraphs = text.split('\n\n')

        segment_headings = {0: Heading(0, '', self.document_name)}
        segment = Segment(level=0, headings=segment_headings, paragraphs=[]) # preamble

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
            for level in LEVELS: # have to delete the headings at higher levels in case of skips later
                if level in new_headings and level > match.level:
                    del new_headings[level]
            segment = Segment(level=self.state, headings=new_headings, paragraphs=[]) # start a new segment
        return self.document
    
    def summarize_matches(self, text):
        paragraphs = text.split('\n\n')
        for paragraph in paragraphs:
            match = match_heading(paragraph, self.patterns)
            if match:
                print(f"{' ' * 4 * match.level}{match.level} heading: {match.enumeration} {match.heading_text}")

##################################################
## Parsing
## FIXME: consider moving to a separate llm  tools module?

LANGUAGE_MODEL = "gpt-4o"
CONTEXT_WINDOW = 128000

import marvin
from openai import OpenAI

openai_client = OpenAI()

def llm(prompt: str, system: str = "", model: str = "gpt-4o") -> str | None:
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
        model=model,
    )
    return chat_completion.choices[0].message.content

@marvin.fn
def infer_regex(examples: list[str]) -> str: # FIXME: do marvin functions return None on failure?
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

def infer_heading_patterns(example_headings: dict[int, list[str]]) -> dict[int, HeadingPattern]:
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

def infer_level_names(patterns: dict[int, HeadingPattern]) -> dict[int, str]:
    return {k: letters_only(infer_level_name(v)) for k, v in patterns.items()}

def remove_newlines(s: str) -> str:
    return re.sub(r'[\n\r]', ' ', s)

@dataclass
class Jurisdiction:
    name: str
    title: str
    patterns: dict[int, HeadingPattern]
    level_names: dict[int, str]
    source_local: str = ''
    source_url: str = ''
    raw_text: str = ''
    parser: StateMachineParser | None = None
    document: list[Segment] = field(default_factory=list)
    autoload: bool = True
    autoparse: bool = False

    def __post_init__(self):
        if self.autoload:
            self.load()
        if self.autoparse:
            self.parse()

    def load(self):
        """Loads the text of local code from file (source_local)."""
        try:
            with open(self.source_local, "r") as f:
                self.raw_text = f.read()
        except FileNotFoundError as e:
            print(f"Error reading {self.source_local}: {e}")

    def parse(self):
        self.parser = StateMachineParser(f"{self.name} Code", self.patterns)
        self.document = self.parser.parse(self.raw_text)

    def chunkify(self, n: int = 1000):
        for segment in self.document:
            if len(segment.paragraphs) == 0:
                continue
            segment.chunkify(n)

    def summarize(self):
        for segment in self.document:
            if len(segment.paragraphs) == 0:
                continue
            text = '\n'.join(segment.paragraphs)
            heading = segment.headings[segment.level]
            level_name = self.level_names[segment.level]
            print(f"{' ' * 4 * segment.level}H{segment.level} {level_name}", end='')
            print(f" {heading.enumeration} ({remove_newlines(heading.heading_text)})", end='') if heading else print()
            print(f": {len(segment.chunks)} chunks, {len(segment.paragraphs)} paragraphs, {len(text)} characters")

##################################################
## Embeddings

EMBEDDING_MODEL = "text-embedding-3-small"

def create_embedding(text: str, model=EMBEDDING_MODEL) -> list[float]:
    """Create an embedding for a block of text."""
    client = OpenAI()
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

##################################################
## Uploading to database

from psycopg import connect

EMBEDDING_LENGTH = len(create_embedding("test"))

def connection(db: dict):
    return connect(
        dbname=db['dbname'],
        host=db['host'],
        port=db['port'],
        autocommit=True
    )

def fill_in(place: Jurisdiction) -> tuple[dict[int, str], dict[int, HeadingPattern]]:
    """Fill in missing level names and patterns with blanks."""
    level_names = {}
    patterns = {}
    for level in LEVELS:
        if level in place.level_names:
            level_names[level] = place.level_names[level]
        else:
            level_names[level] = ''
        if level in place.patterns:
            patterns[level] = place.patterns[level]
        else:
            patterns[level] = HeadingPattern(level=level, regex='', multi_line=False)
    return level_names, patterns

def upload_code(db: dict, jurisdiction: Jurisdiction) -> None:
    """Upload metadata about the jurisdiction to the `codes` table."""
    level_names, patterns = fill_in(jurisdiction)
    with connection(db) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO codes (title, jurisdiction,
                H1_name, H1_pattern,
                H2_name, H2_pattern,
                H3_name, H3_pattern,
                H4_name, H4_pattern,
                H5_name, H5_pattern)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (jurisdiction) DO NOTHING;
                """,
                (jurisdiction.title, jurisdiction.name,
                level_names[1], patterns[1].regex,
                level_names[2], patterns[2].regex,
                level_names[3], patterns[3].regex,
                level_names[4], patterns[4].regex,
                level_names[5], patterns[5].regex)
            )

def upload_segments(db: dict, jurisdiction: Jurisdiction) -> None:
    """Upload segments of the code to the `segments` table."""
    with connection(db) as conn:
        with conn.cursor() as cursor:
            for segment in jurisdiction.document:
                if not segment.paragraphs:
                    continue # skip empty segments (usually part of a table of contents)
                headings = [segment.headings.get(i, None) for i in LEVELS]
                heading_enumerations = [heading.enumeration if heading else None for heading in headings]
                heading_texts = [heading.heading_text if heading else None for heading in headings]
                cursor.execute(
                    """
                    INSERT INTO segments (code_id, segment_level,
                        H1_enumeration, H1_text,
                        H2_enumeration, H2_text,
                        H3_enumeration, H3_text,
                        H4_enumeration, H4_text,
                        H5_enumeration, H5_text,
                        content)
                    VALUES ((SELECT code_id FROM codes WHERE jurisdiction=%s),
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING segment_id;
                    """,
                    (jurisdiction.name, segment.level,
                    heading_enumerations[1], heading_texts[1],
                    heading_enumerations[2], heading_texts[2],
                    heading_enumerations[3], heading_texts[3],
                    heading_enumerations[4], heading_texts[4],
                    heading_enumerations[5], heading_texts[5],
                    '\n\n'.join(segment.paragraphs))
                )
                retval = cursor.fetchone()
                if retval:
                    segment_id = retval[0]
                else:
                    warn(f"Failed to upload segment {segment.level} for {jurisdiction.name}")
                    continue
                cursor.executemany(
                    """
                    INSERT INTO chunks (segment_id, chunk_idx, content)
                    VALUES (%s, %s, %s)
                    """,
                    [(segment_id, i, chunk) for i, chunk in enumerate(segment.chunks)]
                )

def upload_chunks(db: dict, jurisdiction: Jurisdiction) -> None:
    """Upload chunks and their embedding vectors to the `chunks` table."""
    with connection(db) as conn:
        with conn.cursor() as cursor:
            for segment in jurisdiction.document:
                for i, chunk in enumerate(segment.chunks):
                    cursor.execute(
                        """
                        INSERT INTO chunks (segment_id, chunk_number, content, embedding)
                        VALUES ((SELECT segment_id FROM segments WHERE code_id=(SELECT code_id FROM codes WHERE jurisdiction=%s) AND segment_level=%s),
                        %s, %s, %s)
                        """,
                        (jurisdiction.name, segment.level, i, chunk, create_embedding(chunk))
                    )
    return

def upload(db: dict, jurisdiction: Jurisdiction) -> None:
    """Upload (1) metadata about the jurisdiction, to the `codes` table, (2) segments of the code,
    to the `segments` table, and (3) chunks and their embedding vectors, to the `chunks` table."""
    upload_code(db, jurisdiction)
    upload_segments(db, jurisdiction)
    #upload_chunks(db, jurisdiction)
    return

##################################################
## Associations

CONTEXT_TYPES = {
    'rule': 'Statement of a rule, obligation, or prohibition',
    'penalty': 'Penalties for violations of rules specified in other parts of the code',
    'definition': 'Definition of terms used in other parts of the code',
    'interpretation': 'Guidance about interpretation of language or rules, other than definitions',
    'date': 'Effective dates of rules in the document or enacting legistlation, or termination dates',
    'other': 'Any other type of context not covered by the above categories',
    }

def classify_context(text: str, headings: dict[str, str], model=LANGUAGE_MODEL) -> str | None:
    """Determines the nature of the text among predetermined CONTEXT_TYPES.keys().
    Args:
        text: a block of text taken from a municipal code or ordinance
        headings: a dict of headings from the top of the code hierarchy to the supplied text
        model: the name of the OpenAI language model to use
    Returns:
        A single word classification, or None upon failure
    """
    client = OpenAI()

    heading = next(reversed(list(headings.values()))) # last value in dict

    fmt_string = '\n'.join( [f" * '{k}': {v}" for k, v in CONTEXT_TYPES.items()] )

    system_prompt = f"""
You will be given a text block, which may provide context about other parts of
the document (a municipal code or ordinance) from which it was taken.

Your task is to classify the text block. Your response should be a single word
from the following list, choosing the best fit according to the explanation
provided on each line:

{fmt_string}
"""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{heading}\n{text}"}
        ],
        temperature=0.7, # from OpenAI prompt examples page for similar prompts
        max_tokens=5,
        top_p=1,
    )
    response = response.choices[0].message.content
    if response is not None:
        response = letters_only(response)
        if response in CONTEXT_TYPES:
            return response

    warn(f'LLM FAILED TO CLASSIFY CONTEXT. Response: {response} not in {list(CONTEXT_TYPES.keys())}')
    return None


def context_scope(text: str, headings: dict[str, str], context_type: str, model=LANGUAGE_MODEL) -> str | None:
    """Finds the scope of the supplied context.
    Args:
        text: a block of text taken from a municipal code or ordinance
        headings: a dict of headings from the top of the code hierarchy to the supplied text
        context_type: as from classify_context(...)
        model: the name of the OpenAI language model to use
    Returns:
        A single word scope, from among headings.keys(), or None upon failure
    """
    assert context_type in CONTEXT_TYPES.keys()
    if context_type == 'rule': return 'global'

    client = OpenAI()

    heading = next(reversed(list(headings.values()))) # last value in dict
    hierarchy = ", ".join(list(headings.keys()))
 
    system_prompt = f"""
You will be given a text block, which may provide context about other parts of
the document (a municipal code or ordinance) from which it was taken.

Assume that the text block has been classified as '{context_type}': '{CONTEXT_TYPES[context_type]}'.

Your task is to describe the scope of the context provided by this text block within
the document. The document is organized according to the following section hierarchy: {hierarchy}.
The context may also be globabl, applying to the entire code, regulation, or ordinance.

Your response should be a single-word description of the scope, using the terms 
provided in the section hierarchy above. If the scope is the entire code, regulation, or ordinance,
return 'global'. If the scope includes a range of segments at a particular level of the
hierarchy (e.g., 'sections 5-10'), return the next level up (e.g., 'chapter') to cover all of them.
If the scope is not clear, or the block of text does not appear to relate to
definitions, return 'unclear'.
    
Example: if the text begins 'for the purposes of this Code', the scope is 'global'.

Example: if the text begins 'for any section of this Code', the scope is 'global'
because the context applies to the entire Code.

Example: if the text begins 'for the purposes of this section', the scope is 'section'.
    
Example: if the text begins 'for the purposes of sections A through F', the scope is 'chapter'
(assuming 'chapter' is above 'section' in the hierarchy) because the text covers a range of sections.
"""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{heading}\n{text}"}
        ],
        temperature=0.7, # from OpenAI prompt examples page for similar prompts
        max_tokens=5,
        top_p=1,
    )
    response = response.choices[0].message.content
    if response is not None:
        response = letters_only(response)
        if response in headings or response == 'global':
            return response

    warn(f'LLM FAILED TO CLASSIFY SCOPE. Response: {response} not in {list(headings.keys())}')
    return None


def analyze_context(text: str, headings: dict[str, str], model=LANGUAGE_MODEL) -> tuple[str, str] | None:
    """Given a block of text and headings, determines the type of context and its scope.
    Args:
        text: a block of text
        headings: a dict of headings that define the document hierarchy
        model: the OpenAI language model to use
    Returns:
        A tuple consisting of the type of context (e.g., penalty, effective date, interpretation)
        followed by the scope (section, chapter, etc.). Returns None on failure.
    """
    context_type = classify_context(text, headings, model)
    if context_type is None:
        return None
    scope = context_scope(text, headings, context_type, model)
    if scope is None:
        return None
    return context_type, scope


def is_relevant(text: str, query: str, threshold: int = 4, model: str = LANGUAGE_MODEL) -> bool:
    client = OpenAI()

    system_prompt = """You will be given a text block and a query. Your task is
to determine whether the text block is relevant to the query. Please respond
with a single integer from 1 to 5 (inclusive), where 1 means 'the text is not
related to, and does not help to answer, the query', 5 means 'the text is
clearly and related to, and helps to directly answer, the query', and
intermediate values represent degrees of relevance in between. 
"""
    prompt = f"""
Text block: {text}

Query: {query}

Your response (1, 2, 3, 4, or 5):
"""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=2,
        top_p=1
    )
    response = response.choices[0].message.content
    if response is not None:
        try:
            if int(response) >= threshold: return True
        except ValueError as e:
            print(f'WARNING: cannot convert response {response} to integer. Error: {e}')
    
    return False

# Go through rows in the muni database and identify definitions

sql_select = """
    SELECT  id,
        L1_ref, L1_heading,
        L2_ref, L2_heading,
        L3_ref, L3_heading,
        L4_ref, L4_heading,
        text
    FROM muni;
    """

sql_unique = """
    BEGIN
        IF NOT EXISTS (
            SELECT FROM pg_constraint
            WHERE conname = 'unique_associations')
            AND   conrelid = 'muni_associations'::regclass
        ) 
        THEN
            ALTER TABLE muni_associations
            ADD CONSTRAINT unique_associations UNIQUE (jurisdiction, association, left_id, right_id);
        END IF;
    END;
    """

sql_assoc = """
    INSERT INTO muni_associations (jurisdiction, association, left_id, right_id)
    VALUES (%s, %s, %s, %s)
    ON CONFLICT (jurisdiction, association, left_id, right_id) DO NOTHING;
    """

def scope_map(scope):
    """For a given scope, what are the columns in muni that need to match?"""
    table = {'global': ['jurisdiction'],
             'title': ['jurisdiction', 'L1_ref'],
             'chapter': ['jurisdiction', 'L1_ref', 'L2_ref'],
             'article': ['jurisdiction', 'L1_ref', 'L2_ref', 'L3_ref'],
             'section': ['jurisdiction', 'L1_ref', 'L2_ref', 'L3_ref', 'L4_ref']
             }
    if scope not in table.keys():
        return None
    return table[scope]


def set_associations(conn, id_, scope, context_type):
    """Set associations with a row in muni with all rows matching the scope.
    Args:
        conn: a connection to the database
        id_: the id of the row to associate
        scope: the scope of the association (e.g. 'title', 'chapter', 'article', 'section')
        context_type: the type of association (e.g. 'definition')
    """
    with conn.cursor() as cursor:
        # get the jurisdiction and the references
        cursor.execute(f"SELECT jurisdiction, L1_ref, L2_ref, L3_ref, L4_ref FROM muni WHERE id = {id_}")
        jurisdiction, L1_ref, L2_ref, L3_ref, L4_ref = cursor.fetchone()
        # get the columns that need to match
        columns = scope_map(scope)
        if not columns:
            return
        # get the rows that match the scope
        match_str = ' AND '.join([f"{col} = '{val}'" for col, val in zip(columns, [jurisdiction, L1_ref, L2_ref, L3_ref, L4_ref])])
        cursor.execute(f"SELECT id FROM muni WHERE {match_str} AND id != {id_}")
        rows = cursor.fetchall()
        # set the associations
        for row in rows:
            cursor.execute(sql_assoc, (jurisdiction, context_type, id_, row[0]))

def find_associations(conn, jurisdiction):
    return
    ## FIXME: update for new schema
    allowed_types = ['penalty', 'definition', 'interpretation', 'date']
    with conn.cursor() as cursor:
        cursor.execute(sql_select)
        rows = cursor.fetchall()
        for row in rows:
            id_, L1_ref, L1_heading, L2_ref, L2_heading, L3_ref, L3_heading, L4_ref, L4_heading, text = row
            headings = {'title': L1_heading, 'chapter': L2_heading, 'article': L3_heading, 'section': L4_heading}
            r = analyze_context(text, headings, model='gpt-4')
            if r:
                context_type, scope = r
                if context_type in allowed_types:
                    print(f"* Setting associations for id {id_}")
                    print(f"  Context type: {context_type}; Scope: {scope}")
                    print("  --> %s ..." % text[:80].replace('\n', ' '))
                    set_associations(conn, id_, scope, context_type)

##################################################
## Querying

def simple_semantic_query(conn, query, limit=10):
    query_embedding = create_embedding(query)
    with conn.cursor() as cursor:
        sql = """
        SELECT id, L4_heading, text
        FROM muni
        WHERE jurisdiction = 'Chicago'
        ORDER BY embedding <=> %s
        LIMIT %s;
        """
        cursor.execute(sql, (str(query_embedding), limit))
        return cursor.fetchall()
        
def simple_full_text_query(conn, query, limit=10):
    with conn.cursor() as cursor:
        sql = """
        WITH tsq AS (
            SELECT to_tsquery('english', %s) AS search
            )
        SELECT id, L4_heading, text
        FROM muni, tsq
        WHERE jurisdiction = 'Chicago'
        AND textsearchable @@ tsq.search
        ORDER BY ts_rank_cd(textsearchable, tsq.search)
        LIMIT %s;
        """
        cursor.execute(sql, (query, limit))
        return cursor.fetchall()

# Now we do a more complicated hybrid search, borrowing and adapting from 
# https://github.com/pgvector/pgvector-python/blob/master/examples/hybrid_search_rrf.py

def hybrid_query(conn, query, limit=10):
    embedding = create_embedding(query)

    sql = """
    WITH semantic_search AS (
        SELECT id, L4_heading, RANK () OVER (ORDER BY embedding <=> %(embedding)s) AS rank
        FROM muni
        ORDER BY embedding <=> %(embedding)s
        LIMIT 20
    ),
    keyword_search AS (
        SELECT id, L4_heading, RANK () OVER (ORDER BY ts_rank_cd(textsearchable, query) DESC)
        FROM muni, plainto_tsquery('english', %(query)s) query
        WHERE textsearchable @@ query
        ORDER BY ts_rank_cd(textsearchable, query) DESC
        LIMIT 20
    )
    SELECT
        COALESCE(semantic_search.id, keyword_search.id) AS id,
        COALESCE(1.0 / (%(k)s + semantic_search.rank), 0.0) +
        COALESCE(1.0 / (%(k)s + keyword_search.rank), 0.0) AS score,
        COALESCE(semantic_search.L4_heading, keyword_search.L4_heading) AS L4_heading
    FROM semantic_search
    FULL OUTER JOIN keyword_search ON semantic_search.id = keyword_search.id
    ORDER BY score DESC
    LIMIT %(limit)s;
    """
    result = conn.execute(sql, {'query': query, 'embedding': str(embedding), 'limit': limit, 'k': 60})
    return result.fetchall()
