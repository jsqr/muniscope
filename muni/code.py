from dataclasses import dataclass, field
from copy import deepcopy
import re
from typing import Any

###################
# Parsing code text
# This version uses hard-coded regular expressions for scanning headings
# in the document hierarchy, and (essentiall) a hand-coded recursive
# descent parser to extract the text into a tree structure.

@dataclass
class Node():
    label: str
    ref: str = ''
    heading: str = ''
    text: str = ''
    metadata: dict[str, Any] = field(default_factory=dict)
    children: 'list[Node]' = field(default_factory=list)

@dataclass
class Jurisdiction:
    name: str
    hierarchy: dict[str, str]
    source_local: str = ''
    source_url: str = ''
    text: str = ''
    autoload: bool = True    

    def __post_init__(self):
        if self.autoload:
            self.load()

    def load(self):
        """Loads the text of local code from file (source_local)."""
        try:
            with open(self.source_local, "r") as f:
                self.text = f.read()
        except FileNotFoundError as e:
            print(f"Error reading {self.source_local}: {e}")

    def parse(self):
        """Parses the text of a jurisdiction into a tree of Nodes."""
        node = Node(
            label="top",
            ref="",
            heading="",
            text=self.text,
            metadata={"jurisdiction": self.name,
                      "source_url": self.source_url,
                      "source_local": self.source_local,
                      "references": {},
                      "headings": {},
                      },
            children=[],
        )
        return parse_rec(node, self.hierarchy)
    
def parse_rec(node, hierarchy):
    """Recursively parse text according to a hierarchy of patterns."""
    assert len(hierarchy) > 0, "Hierarchy must have at least one level"

    first, rest_hierarchy = first_keyval_and_rest(hierarchy)
    label, pattern = first.popitem()
    splitting_pattern = r'(^|\n\n)' + pattern # only match the beginning of a paragraph
    segments = split_text(splitting_pattern, node.text)
    
    # Handle the case where the code skips a level in the hierarchy (it happens sometimes)
    if len(segments) == 1 and rest_hierarchy:
        ref, rest = match_and_rest(pattern, segments[0])
        if not ref:
            return parse_rec(node, rest_hierarchy)

    node.text = '' # clear the text of the current node
    children = []
    for segment in segments:
        ref, rest = match_and_rest(pattern, segment)
        if not ref:
            node.text = segment.strip() # initial 'preamble' text gets added to current node
            continue
        heading, rest = first_paragraph_and_rest(rest)
    
        child_metadata = deepcopy(node.metadata)
        child_metadata["references"][label] = ref
        child_metadata["headings"][label] = heading

        child = Node(
            label=label,
            ref=ref,
            heading=heading,
            metadata=child_metadata,
            text=rest.strip(),
            children=[]
            )
        if rest_hierarchy:
            child = parse_rec(child, rest_hierarchy)

        children.append(child)

    node.children = children
    return node

###################
# Parsing utilities

def first_paragraph_and_rest(text):
    """Extract the first paragraph from the rest of the text"""
    match = re.match(r"\s*(.*?)(?:\n\n|$)", text, re.DOTALL)
    if match:
        return match.group(1), text[match.end():]
    return "", text
    
def first_keyval_and_rest(dictionary):
    """Splits a dictionary into its first key-value pair and the rest of the dictionary."""
    key = next(iter(dictionary))
    first_keyval = {key: dictionary[key]}
    rest = {k: v for k, v in dictionary.items() if k != key}
    return first_keyval, rest

def match_and_rest(pattern, text):
    """Returns the first match and the rest of the text"""
    match = re.search(pattern, text)
    if match:
        return match.group(), text[match.end():]
    return None, text

def split_text(pattern, text):
    """Splits text into segments based on a pattern. The matched pattern is
    included in the following segment. Doesn't include empty segments."""
    segments = []
    start = 0
    for match in re.finditer(pattern, text):
        end = match.start()
        segments.append(text[start:end])
        start = match.start()
    segments.append(text[start:])
    return [s for s in segments if s]

def split_into_blocks(text, max_chars):
    """Splits text into blocks of at most `max_chars`, avoiding splitting paragraphs."""
    if len(text) <= max_chars:
        return [text.strip()]

    paragraphs = text.split('\n\n')
    paragraphs = [p.strip() for p in paragraphs]

    segments = []
    current_segment = ''
    current_chars = 0

    for p in paragraphs:
        if current_chars + len(p) > max_chars and current_segment:
            # start a new segment
            segments.append(current_segment)
            current_segment = p
            current_chars = len(p)
        elif p:
            # add to current segment
            current_segment = '\n\n'.join([current_segment, p]) if current_segment else p
            current_chars += len(p)

    # add the last segment
    if current_segment:
        segments.append(current_segment)

    return segments
