from openai import OpenAI
from typing import Any
from warnings import warn

# LLM functions for generating embeddings, summaries, and context analysis
#
# Uses the OpenAI API to create embeddings and generate summaries. You will 
# need to set the OPENAI_API_KEY environment variable to your API key. One way
# to do this is to have a .env file in the root director with a line like this:
#
# OPENAI_API_KEY=sk-YOUR_API_KEY
#
# IMPORTANT: Make sure to add .env to your .gitignore file so that you don't
# accidentally commit your API key to a public repository

EMBEDDING_MODEL = "text-embedding-3-small"
LANGUAGE_MODEL = "gpt-3.5-turbo-0125"
CONTEXT_WINDOW = 16000
MAX_SUMMARY_SIZE = 1000

#
# LLM wrapper functions
#

def create_embedding(text: str, model=EMBEDDING_MODEL) -> list[float]:
    """Create an embedding for a block of text."""
    client = OpenAI()
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

AUGMENT_SYSTEM_PROMPT = """Write a provision from a municipal code or ordinance that would be
responsive to the query below. If the query asks whether the code or ordinance has
a provision on a topic, you should assume that it does. If the query asks for
details, provide them in a plausible way."""

AUGMENT_PROMPT = """Query: {} 
Code provision: """

def augmented_embedding(query: str, system_prompt=AUGMENT_SYSTEM_PROMPT, prompt: str=AUGMENT_PROMPT,
                        num_samples=5, orig_weight=0.1, model=LANGUAGE_MODEL) -> list[float]:
    """Augment the query by first generating a series of hypothetical results,
    then creating embeddings for each, and finally averaging the results (HyDE).
    Returns a weighted average of the original query embedding and the HyDE average."""
    assert 0 <= orig_weight <= 1.0
    prompt = prompt.format(query)

    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7, # from OpenAI prompt examples page for similar prompts
        max_tokens=1000,
        top_p=1,
        n=num_samples,
    )

    embeddings = []
    for choice in response.choices:
        hypothetical = choice.message.content
        if hypothetical is None:
            continue
        embeddings.append(create_embedding(hypothetical))

    orig_embedding = create_embedding(query)
    orig_embedding_wt = vector_mult(orig_embedding, orig_weight)
    # Compute the average of the embeddings for generated hypothetical code fragments
    hyde_embedding = vector_ave(embeddings)
    hyde_embedding_wt =  vector_mult(hyde_embedding, 1.-orig_weight)   

    return vector_sum([orig_embedding_wt, hyde_embedding_wt])

# FIXME: This is absurd. Convert to numpy?
def transpose(vs: list[list[float]]) -> list[list[float]]:
    return list(map(list, zip(*vs)))

def vector_ave(v: list[list[float]]) -> list[float]:
    return [sum(group) / len(group) for group in transpose(v)]

def vector_sum(vs: list[list[float]]) -> list[float]:
    return list(map(sum, zip(*vs)))

def vector_mult(v: list[float], b: float) -> list[float]:
    return [b*x for x in v]


def summarize_short(text: str, model=LANGUAGE_MODEL, max_length=MAX_SUMMARY_SIZE) -> str | None:
    """Summarizes a block of text no more than CONTEXT_WINDOW - MAX_SUMMARY_SIZE characters."""
    
    if len(text) > CONTEXT_WINDOW - MAX_SUMMARY_SIZE:
        warn(f"WARNING: text is too long for the model context window ({len(text)} > {CONTEXT_WINDOW - MAX_SUMMARY_SIZE})")
        text = text[:CONTEXT_WINDOW - MAX_SUMMARY_SIZE]

    client = OpenAI()

    system_prompt = f"""
You will be given a block of text, and your task is to accurately summarize the
text in as much detail as possible without exceeding {max_length} characters.
"""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ],
        temperature=0.7, # from OpenAI prompt examples page for similar prompts
        max_tokens=max_length,
        top_p=1,
    )

    return response.choices[0].message.content


def summarize(text: str, model=LANGUAGE_MODEL, max_length=MAX_SUMMARY_SIZE) -> str | None:
    """Summarize the text. If the text is too long for the model context window,
     break it into chunks, summarize each chunk, then summarize the summaries."""

    if len(text) <= MAX_SUMMARY_SIZE:
        return text

    # bookkeeping to stay within the model's context window
    input_budget = CONTEXT_WINDOW - MAX_SUMMARY_SIZE
    approx_num_chunks = len(text) // input_budget + 1
    chunk_summary_budget = input_budget // approx_num_chunks
    chunk_summary_budget = min(MAX_SUMMARY_SIZE, chunk_summary_budget) # avoid output token limit issues
    chunk_budget = CONTEXT_WINDOW - chunk_summary_budget

    result = ""
    while len(text) > chunk_budget:
        # Find the last period before the context window
        last_period = text[:chunk_budget].rfind(".")
        if last_period == -1:
            last_period = chunk_budget
        else:
            last_period += 1
        # Summarize the text up to the last period
        summary = summarize_short(text[:last_period], model, chunk_summary_budget)
        # Remove the summarized text from the original text
        text = text[last_period:]
        # Add the summarized text to the result
        if summary is None:
            return None
        result += summary

    return summarize_short(result, model, max_length)


def clean(s: str) -> str:
    return ''.join([c for c in s if c.isalpha()])

def definition(text: str, headings: dict[str, str], model=LANGUAGE_MODEL) -> str | None:
    """Given a block of text and headings, return the scope of definitions (or None).
    Args:
        text: a block of text
        headings: a dict  of headings that define the document hierarchy
        model: the OpenAI language model to use
    Returns:
        The scope of any definitions, 'unclear' if the scope is unclear, or None if
        the supplied headings do not indicate that definitions are present.
    """
    client = OpenAI()

    hierarchy = ", ".join(list(headings.keys()))
    is_def = any(['definition' in h.lower() for h in headings.values()])
    if not is_def:
        return None

    system_prompt = f"""
You will be given a block of text, which may introduce
a set of definitions. Your task is to identify the scope of the definitions
within the document. The document is organized according to the following
hierarchy: {hierarchy}. The defininions may also apply to the entire code,
regulation, or ordinance.
    
Please provide a single-word description of the scope, using the terms 
provided above. If the scope is the entire code, regulation, or ordinance,
return 'global'. If the scope includes multiples at a particular level of the
hierarchy (e.g., multiple sections), return the next level up (e.g., 'chapter').
If the scope is not clear, or the block of text does not appear to relate to
definitions, return 'unclear'.
    
Example: if the text begins 'for the purposes of this section', the scope is 'section'.
    
Example: if the text begins 'for the purposes of sections A through F', the scope is 'chapter'
(assuming 'chapter' is above 'section' in the hierarchy).
"""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ],
        temperature=0.7, # from OpenAI prompt examples page for similar prompts
        max_tokens=2,
        top_p=1,
    )
    response = response.choices[0].message.content
    if response is not None:
        if clean(response) in 'global, ' + hierarchy:
            return response
    return None


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
        response = clean(response)
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
        response = clean(response)
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

#
# UTILITIES
#

def last_value(d: dict[Any, Any]) -> Any:
    return next(reversed(list(d.values())))
