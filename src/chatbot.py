import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch


# -----------------------------
# Page settings
# -----------------------------

st.set_page_config(
    page_title="Hindu Marriage Act Legal Chatbot",
    page_icon="⚖️"
)
# -----------------------------
# Chat history
# -----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []
# -----------------------------
# Sidebar
# -----------------------------

with st.sidebar:

    st.header("⚖️ Legal Chatbot")

    if st.button("🗑️ Clear Chat"):

        st.session_state.messages = []

        st.rerun()
st.title("⚖️ Hindu Marriage Act Legal Chatbot")

st.write(
    "Ask questions about the Hindu Marriage Act, 1955. "
    "The chatbot retrieves relevant sections from the Act."
)
st.info(
    "  This chatbot provides information based on the "
    "Hindu Marriage Act, 1955. It is for educational purposes "
    "only and does not constitute legal advice."
)

# -----------------------------
# Load legal text
# -----------------------------

with open(
    "data/hindu_marriage_act.txt",
    "r",
    encoding="utf-8"
) as file:
    text = file.read()


# -----------------------------
# Divide Act into sections
# -----------------------------

pattern = r"(?m)^\s*(\d{1,2}[A-Z]?)\.\s+"

matches = list(re.finditer(pattern, text))

chunks = []

for i, match in enumerate(matches):

    start = match.start()

    if i + 1 < len(matches):
        end = matches[i + 1].start()
    else:
        end = len(text)

    section = text[start:end].strip()

    # Remove chapter headings accidentally included at the end
    section = re.sub(
        r"\n[A-Z][A-Z\s,&\-]{10,}\s*$",
        "",
        section
    ).strip()

    if len(section) > 50:
        chunks.append(section)


# -----------------------------
# Load embedding model
# -----------------------------

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


model = load_model()


# -----------------------------
# Load question-answering model
# -----------------------------

@st.cache_resource
def load_qa_model():

    tokenizer = AutoTokenizer.from_pretrained(
        "deepset/roberta-base-squad2"
    )

    model = AutoModelForQuestionAnswering.from_pretrained(
        "deepset/roberta-base-squad2"
    )

    return tokenizer, model


tokenizer, qa_model = load_qa_model()


# -----------------------------
# Create FAISS index
# -----------------------------

@st.cache_resource
def create_index():

    embeddings = model.encode(chunks)

    embeddings = np.array(
        embeddings
    ).astype("float32")

    index = faiss.IndexFlatL2(
        embeddings.shape[1]
    )

    index.add(embeddings)

    return index


index = create_index()


# -----------------------------
# Search legal sections
# -----------------------------

def search_legal_text(question, top_k=3):

    question_lower = question.lower()

    # Handle common spelling mistake
    question_lower = question_lower.replace(
        "marraige",
        "marriage"
    )

    # ---------------------------------
    # Special search: marriage age
    # ---------------------------------

    if (
        re.search(r"\bage\b", question_lower)
        and "marriage" in question_lower
    ):

        for chunk in chunks:

            chunk_lower = chunk.lower()

            if (
                "twenty-one years" in chunk_lower
                and "eighteen years" in chunk_lower
            ):
                return [chunk]


    # ---------------------------------
    # Special search: restitution
    # ---------------------------------

    if (
        "restitution" in question_lower
        and "conjugal" in question_lower
    ):

        for chunk in chunks:

            if "9. Restitution of conjugal" in chunk:

                return [chunk]

    # ---------------------------------
    # Special search: voidable marriage
    # ---------------------------------

    if "voidable marriage" in question_lower:

        for chunk in chunks:

            chunk_lower = chunk.lower()

            if (
                chunk.startswith("12.")
                and "voidable" in chunk_lower
            ):
                return [chunk]
    # ---------------------------------
    # Special search: divorce
    # ---------------------------------

    if "divorce" in question_lower:

        for chunk in chunks:

            chunk_lower = chunk.lower()

            # Section 13 contains the grounds for divorce
            if (
                chunk.startswith("13.")
                and "divorce" in chunk_lower
            ):
                return [chunk]


    # ---------------------------------
    # Normal FAISS search
    # ---------------------------------

    question_embedding = model.encode(
        [question]
    )

    question_embedding = np.array(
        question_embedding
    ).astype("float32")

    distances, indices = index.search(
        question_embedding,
        top_k
    )

    results = []

    for i in indices[0]:

        if 0 <= i < len(chunks):

            if chunks[i] not in results:
                results.append(chunks[i])

    return results


    # ---------------------------------
    # Special search for marriage age
    # ---------------------------------

    if (
        "age" in question_lower
        and "marriage" in question_lower
    ):

        for chunk in chunks:

            chunk_lower = chunk.lower()

            if (
                "twenty-one years" in chunk_lower
                and "eighteen years" in chunk_lower
            ):

                return [chunk]


    # Remove duplicates
    results = []

    for result in faiss_results:

        if result not in results:
            results.append(result)

    return results


# -----------------------------
# Generate answer
# -----------------------------

def generate_answer(question, results):

    question_lower = question.lower()

    # Handle common spelling mistake
    question_lower = question_lower.replace(
        "marraige",
        "marriage"
    )

    # ---------------------------------
    # Special answer: marriage age
    # ---------------------------------

    if (
        re.search(r"\bage\b", question_lower)
        and "marriage" in question_lower
    ):

        return (
            "Under Section 5(iii) of the Hindu Marriage Act, "
            "the bridegroom must have completed 21 years of age "
            "and the bride must have completed 18 years of age "
            "at the time of marriage."
        )

    # ---------------------------------
    # Special answer: restitution
    # ---------------------------------

    if (
        "restitution" in question_lower
        and "conjugal" in question_lower
    ):

        return (
            "Restitution of conjugal rights means that when a "
            "husband or wife has withdrawn from the society of "
            "the other without reasonable excuse, the aggrieved "
            "party may apply to the district court for restitution "
            "of conjugal rights under Section 9 of the Hindu "
            "Marriage Act."
        )

    # ---------------------------------
    # Special answer: void marriage
    # ---------------------------------

    if "void marriage" in question_lower:

        return (
            "Under Section 11 of the Hindu Marriage Act, a void "
            "marriage is a marriage that is null and void if it "
            "contravenes any one of the conditions specified in "
            "clauses (i), (iv), or (v) of Section 5."
        )
    
    # ---------------------------------
    # Special answer: voidable marriage
    # ---------------------------------

    if "voidable marriage" in question_lower:

        return (
            "Under Section 12 of the Hindu Marriage Act, "
            "a voidable marriage is a marriage that can be "
            "declared void by a decree of nullity on specified "
            "grounds provided under the Act."
        )
    
    
    # ---------------------------------
    # Special answer: bigamy
    # ---------------------------------

    if (
        "bigamy" in question_lower
        or (
            "marries again" in question_lower
            and "spouse" in question_lower
        )
        or (
            "marry again" in question_lower
            and "spouse" in question_lower
        )
    ):

        return (
            "Under Section 17 of the Hindu Marriage Act, a marriage "
            "between two Hindus is void if, at the time of the marriage, "
            "either party has a husband or wife living. The section "
            "also provides that the provisions relating to bigamy "
            "apply accordingly."
        )
        # ---------------------------------
    # Special answer: marriage registration
    # ---------------------------------

    if (
        "registration" in question_lower
        and "marriage" in question_lower
    ):

        return (
            "Under Section 8 of the Hindu Marriage Act, registration "
            "of a Hindu marriage may be made compulsory by the State "
            "Government. Therefore, whether registration is compulsory "
            "depends on the rules applicable in the particular State."
        )
    # ---------------------------------
    # Special answer: grounds for divorce
    # ---------------------------------

    if "divorce" in question_lower:

        return (
            "Under Section 13 of the Hindu Marriage Act, grounds "
            "for divorce include adultery, cruelty, desertion for "
            "at least two years, conversion from Hinduism, certain "
            "forms of mental disorder, and other grounds specified "
            "in Section 13."
        )
   
    # ---------------------------------
    # General legal questions
    # ---------------------------------

    if not results:

        return (
            "I could not find a relevant section "
            "of the Hindu Marriage Act."
        )

    context = "\n\n".join(results)

    inputs = tokenizer(
        question,
        context,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():

        outputs = qa_model(
            **inputs
        )

    start_index = torch.argmax(
        outputs.start_logits
    )

    end_index = torch.argmax(
        outputs.end_logits
    )

    if end_index < start_index:

        return (
            "The answer could not be found "
            "in the retrieved sections."
        )

    answer_tokens = inputs["input_ids"][0][
        start_index:end_index + 1
    ]

    answer = tokenizer.decode(
        answer_tokens,
        skip_special_tokens=True
    ).strip()

    if not answer:

        return (
            "The answer could not be found "
            "in the retrieved sections."
        )

    return answer

# -----------------------------
# Clean source text
# -----------------------------

def clean_source(source):

    source = re.sub(
        r"\\+",
        "",
        source
    )

    source = re.sub(
        r"\*+",
        "",
        source
    )

    source = re.sub(
        r"\[\d+\]",
        "",
        source
    )

    source = re.sub(
        r"\s+",
        " ",
        source
    )

    return source.strip()


# -----------------------------
# Chat interface
# -----------------------------

# Display previous messages

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.write(message["content"])

question = st.chat_input(
    "Ask a question about the Hindu Marriage Act..."
)


if question:

    st.chat_message(
        "user"
    ).write(question)
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    results = search_legal_text(
        question
    )


    with st.chat_message("assistant"):

        answer = generate_answer(
            question,
            results
        )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        st.write("### Answer")

        st.write(answer)


        st.write("### 📚 Source")

    if results:

        source_text = clean_source(
            results[0]
        )

        source_match = re.match(
            r"(\d+[A-Z]?)\.\s*([^—.-]+)",
            source_text
        )

        if source_match:

            section_number = source_match.group(1)
            section_title = source_match.group(2).strip()

            st.success(
                f"Section {section_number}: {section_title}"
            )

        with st.expander("View legal text"):

            st.write(
                source_text[:2000]
            )

    else:

        st.write(
            "No relevant section found."
        )