from muni import code

test_text = """
Preamble

We hold this test example to be self-evident.

TITLE 1
Heading T1

CHAPTER 1-1 Subheading C1-1

Some text in ch. 1-1.

1-1-1 Section 1-1-1

apples

1-1-2 Section 1-1-2

pears

CHAPTER 1-2 Subheading C1-2

Some more text in ch. 1-2

ARTICLE I. Article in hierarchy!

1-2-1 Section 1-2-1

bananas

1-2-2 Section 1-2-2

grapes

TITLE 2 Heading T2

Further text to conclude the example.
"""

test_jurisdiction = code.Jurisdiction(
    name="Testville",
    hierarchy={
        "title":   r"TITLE \d+",
        "chapter": r"CHAPTER \d+-\d+",
        "article": r"ARTICLE [IVX]+",
        "section": r"\d+-\d+-\d+",
    },
    text=test_text,
    autoload=False
)

def test_parse():
    test_tree = test_jurisdiction.parse()
    assert test_tree.metadata['jurisdiction'] == "Testville"
    assert test_tree.text[:8] == "Preamble"
    assert test_tree.children[0].label == "title"
    assert test_tree.children[0].ref == "TITLE 1"
    assert test_tree.children[0].heading == "Heading T1"
    assert test_tree.children[0].children[0].label == "chapter"
    assert test_tree.children[0].children[0].ref == "CHAPTER 1-1"
    assert test_tree.children[0].children[0].heading == "Subheading C1-1"
    assert test_tree.children[0].children[0].children[0].label == "section"
    assert test_tree.children[0].children[0].children[0].ref == "1-1-1"
    assert test_tree.children[0].children[0].children[0].text[:6] == "apples"
    assert test_tree.children[0].children[1].label == "chapter"
    assert test_tree.children[0].children[1].ref == "CHAPTER 1-2"
    assert test_tree.children[0].children[1].heading == "Subheading C1-2"
    assert test_tree.children[0].children[1].children[0].label == "article"
    assert test_tree.children[0].children[1].children[0].ref == "ARTICLE I"
    assert test_tree.children[0].children[1].children[0].children[0].label == "section"
    assert test_tree.children[1].label == "title"
    assert test_tree.children[1].ref == "TITLE 2"
    assert test_tree.children[1].heading == "Heading T2"
    assert test_tree.children[1].text[:7] == "Further"

def test_first_paragraph_and_rest():
    test1 = "\n\na\nb\n\nc\n"
    test2 = "a\nb\n\nc\n"
    assert code.first_paragraph_and_rest(test1) == ('a\nb', 'c\n')
    assert code.first_paragraph_and_rest(test2) == ('a\nb', 'c\n')

def test_first_keyval_and_rest():
    test = {"a": 1, "b": 2, "c": 3}
    assert code.first_keyval_and_rest(test) == ({"a": 1}, {"b": 2, "c": 3})

def test_match_and_rest():
    test = "T 1 h\n"
    assert code.match_and_rest(r"(T \d+)", test) == ("T 1", " h\n")

def test_split_text():
    test1 = "p\nT 1\na b\nT 2\n"
    test2 = "T 1\na b\nT 2\nc\nT 3\n"
    assert code.split_text(r"T \d+", test1) == ['p\n', 'T 1\na b\n', 'T 2\n']
    assert code.split_text(r"T \d+", test2) == ['T 1\na b\n', 'T 2\nc\n', 'T 3\n']

# def split_into_blocks():
# TODO: write test for code.split_into_blocks