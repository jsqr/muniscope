from muni.code import split_paragraph, match_heading, HeadingPattern, Heading, Level

def test_split_paragraph():
    assert split_paragraph("") == ("", "")
    assert split_paragraph("This is a\nparagraph.\n") == ("This is a", "paragraph.\n")

def test_match_heading():
    test_doc1 = "Chapter VII: The Final Chapter"
    test_pattern1 = HeadingPattern(level=Level.H1, regex=r'^Chapter ([IVXLC]+): (.+)$', multi_line=False)

    test_doc2 = "\n\nChapter 7:\nThe Final Chapter"
    test_pattern2 = HeadingPattern(level=Level.H1, regex=r'^Chapter (\d+):', multi_line=True)

    assert match_heading(test_doc1, {Level.H1: test_pattern1}) == \
        Heading(level=Level.H1, enumeration="VII", heading_text="The Final Chapter")

    assert match_heading(test_doc2, {Level.H1: test_pattern2}) == \
        Heading(level=Level.H1, enumeration="7", heading_text="The Final Chapter")