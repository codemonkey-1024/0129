def prompt_for_KG_contruction_refine(input_text):
    prompt = f"""
--Objective--
Analyze the given text to extract structured information about entities and their relationships.

--Entity Extraction Rules--
1. Identify explicit entities with:
 - PER (Person): Full names or unambiguous references to individuals.
 - ORG (Organization): Names of companies, institutions, or groups.
 - GPE (Geo-Political Entity): Countries, cities, states.
 - LOC (Location): Geographical entities (mountains, rivers).
 - FACILITY: Human-made structures (buildings, airports).
 - EVENT: Named events or conferences.
 - PRODUCT: Commercial products or services.
 - WORK_OF_ART: Titles of creative works (books, movies, art).
 - LAW: Legal documents, regulations, or treaties.
 - LANGUAGE: Names of languages or dialects.
 - DISEASE: Medical conditions or symptoms.
 - DATE: Specific calendar dates.
 - NUMERIC: Numerical values (quantities, percentages).
 - Other: Any additional or domain-specific entities not covered above.

2. For each entity:
   - Use exact textual mentions
   - Merge coreferent mentions (e.g., "Dr. Smith" and "the Professor")
   - Avoid inferences not explicitly stated
   - Contextual Description: A concise phrase summarizing the entity's role or attributes in the text.
   - Format: [entity | ID | Type | Exact Mention | Contextual Description]

--Relationship Extraction Rules--
1. Identify only explicitly stated relationships with:
   - Action verbs indicating interaction
   - Spatial/temporal prepositions
   - Possessive constructions
   - Membership indicators

2. Relationship requirements:
   - Both entities must be previously identified
   - Use canonical verb forms (active voice)
   - Maintain original text directionality
   - Evidence Span: The shortest contiguous text span that supports the relationship.
   - Format: [relation | SourceID | RelationType | TargetID | EvidenceSpan]


--Validation Rules--
1. Strictly prohibit:
   - External knowledge inclusion
   - Negated relationships
   - Hypothetical statements

--Examples--
Text: "Dr. Smith, CEO of TechCorp, announced the launch of the new product, QuantumX, at the NY Summit on May 5, which was attended by 500 participants."
Entities:
[entity | E1 | PER | "Dr. Smith" | CEO of TechCorp]
[entity | E2 | ORG | "TechCorp" | Company employing Dr. Smith]
[entity | E3 | PRODUCT | "QuantumX" | New product launched by TechCorp]
[entity | E4 | EVENT | "NY Summit" | Event attended by 500 participants]
[entity | E5 | DATE | "May 5" | Specific date mentioned]
[entity | E6 | NUMERIC | "500" | Number of participants]
Relations:
[relation | E2 | employs | E1 | "CEO of TechCorp"]
[relation | E1 | announced | E3 | "announced the launch of the new product"]
[relation | E4 | launched | E3 | "launch of the new product at the NY Summit"]
[relation | E4 | occurs on | E5 | "on May 5"]
[relation | E4 | attended by | E6 | "attended by 500 participants"]



--Attention--
 - Output results should strictly match those in the Examples.

-Real Data-
Text: 
"{input_text}"

Output:
    """
    return prompt


def prompt_for_KG_contruction_auto_type_short_(input_text):
    prompt = f"""
--Objective--
Extract entities and their relationships from the given text with strict EID consistency and **no ambiguous pronouns/nouns**.

--Entity Extraction Rules--
1. Assign sequential EIDs starting from E1:
   - Maintain exact case from the text.
   - **Merge coreferences and resolve ambiguous terms (e.g., "he", "she", "it", "the director", "the film") to their explicit antecedents**.
   - If an ambiguous term (e.g., "they", "their") has no clear antecedent, omit it from entity extraction.
2. Entity types must be **specific and domain-appropriate** (e.g., "Antibody Drug" instead of "Product", "Film Director" instead of "Person"):
   - Types include: People, Organizations, Locations, Events, Time Periods, Products, Concepts, etc.
3. Format entities as:
   [entity | EID | Type | "Entity Name" | Description]

--Relationship Extraction Rules--
1. Relationships must:
   - Reference **only explicit entities** (no pronouns/nouns like "this", "that", "the company").
   - Have direct textual evidence (e.g., "approved" from "FDA approved Beovu®").
2. Format relations as:
   [relation | SourceEID | RelationType | TargetEID | "exact quote"]

--Consistency Checks--
1. EIDs must be sequential without gaps/duplicates.
2. All relationship EIDs must exist in the entity list.
3. **No entities derived from ambiguous terms** (e.g., "director" → replaced with "James Cameron" if contextually clear).

--Processing Steps--
1. Extract explicit entities first (e.g., "Novartis", "Beovu®").
2. Resolve coreferences: Replace ambiguous terms (e.g., "it" → "Beovu®") with their explicit antecedents.
3. Validate relationships: Ensure no pronouns/nouns are used as entities.
4. Assemble final output.

--Note--
Ambiguous terms like "the director" and "their" are replaced with explicit entities ("James Cameron", "Studio") when contextually clear. If no antecedent exists, omit the entity/relationship.


--Example--
Text Input:
"James Cameron's film *Avatar* won an Oscar. The director thanked the studio for their support."

Output:
Entities:
[entity | E1 | Film Director | "James Cameron" | Award-winning filmmaker]
[entity | E2 | Film | "*Avatar*" | Sci-fi movie released in 2009]
[entity | E3 | Organization | "Academy of Motion Picture Arts and Sciences" | Oscar-awarding body]
[entity | E4 | Organization | "Studio" | Film production company]

Relations:
[relation | E1 | directed | E2 | "James Cameron's film"]
[relation | E3 | awarded | E2 | "won an Oscar"]
[relation | E1 | thanked | E4 | "thanked the studio"]

-Real Data-
Text Input:
"{input_text}"

Output:
    """
    return prompt



def prompt_for_KG_contruction_auto_type(input_text):
    prompt = f"""
--Objective--
Analyze the given text to extract structured information about entities and their relationships with strict ID consistency.

--Entity Extraction Rules--
1. Entity Identification:
   - Assign sequential EIDs starting from E1 (E1, E2, E3,...)
   - Maintain exact case from source text
   - Merge references to the same entity before assigning EIDs, and prohibit the use of non-explicit pronouns such as he, she, the film, etc.
   - Format entities as:
     [entity | EID | Type | "Entity Name" | Description]

2. Entity Requirements:
    - Types can include a wide range of categories such as People, Organizations, Locations, Events, Time Periods, Products, Concepts, General Entities, Event Entities, etc.
    - Type must use specific natural categories (e.g., "Medical Device" not "PRODUCT")
    - Include functional context in description

--Relationship Extraction Rules--
1. Validation Requirements:
   - Verify existence of both EIDs in entity list
   - Prohibit relations with unregistered EIDs
   - Require direct textual evidence

2. Format Enforcement:
   - Relation format: [relation | SourceEID | RelationType | TargetEID | "exact quote"]
   - Block relations where EID gap > current entity count

--Consistency Checks--
1. ID Validation:
   - Entity EIDs must form unbroken sequence
   - Prohibit duplicate EIDs
   - Restrict EID creation to entity section

2. Cross-reference Validation:
   - Relationship EIDs must match existing entity EIDs
   - Require bi-directional EID verification

--Error Prevention Measures--
1. ID Generation Protocol:
   Relation IDs prohibited
   String-to-EID conversion required

2. Processing Order:
   1. Full entity extraction
   2. Coreference resolution
   3. Relationship validation
   4. Final output assembly

--Enhanced Example--
--Text Input--
"Novartis received FDA approval for Beovu® antibody in Q2 2023."

Output:
Entities:
[entity | E1 | Company | "Novartis" | Pharmaceutical corporation]
[entity | E2 | Regulatory Agency | "FDA" | US medical approval body]
[entity | E3 | Drug | "Beovu®" | Ophthalmic antibody treatment]
[entity | E4 | Time Period | "Q2 2023" | Second quarter of 2023]

Relations:
[relation | E1 | received approval from | E2 | "received FDA approval"]
[relation | E2 | approved | E3 | "approval for Beovu®"]
[relation | E3 | approved in | E4 | "in Q2 2023"]

--Strict Enforcement--
- Reject relations with invalid EIDs
- Terminate processing on EID mismatch
- Require entity-relation EID parity
- Prohibit special characters in EIDs

--Text Input--
"{input_text}"

Output:
    """
    return prompt


def prompt_for_KG_contruction_auto_type_short(input_text):
    prompt = f"""
--Objective--
Extract entities and their relationships from the given text, ensuring strict EID consistency.

--Entity Extraction Rules--
1. Assign sequential EIDs starting from E1.
   - Maintain exact case from the text.
   - Merge coreferences before assigning EIDs.
2. Use specific, natural entity types (e.g., "Medical Device" not "PRODUCT").
   - Types can include a wide range of categories such as People, Organizations, Locations, Events, Time Periods, Products, Concepts, General Entities, Event Entities, etc.
3. Format entities as:
   [entity | EID | Type | "Entity Name" | Description]

--Relationship Extraction Rules--
1. Ensure both EIDs exist in the entity list.
   - Relations must have direct textual evidence.
2. Format relations as:
   [relation | SourceEID | RelationType | TargetEID | "exact quote"]

--Consistency Checks--
1. EIDs must be sequential without gaps or duplicates.
   - EID creation only in the entity section.
2. Relationship EIDs must match existing entity EIDs.

--Processing Steps--
1. Extract all entities.
2. Resolve coreferences.
3. Validate relationships.
4. Assemble final output.

--Example--
Text Input:
"Novartis received FDA approval for Beovu® antibody in Q2 2023."

Output:
Entities:
[entity | E1 | Company | "Novartis" | Pharmaceutical corporation]
[entity | E2 | Regulatory Agency | "FDA" | US medical approval body]
[entity | E3 | Drug | "Beovu®" | Ophthalmic antibody treatment]
[entity | E4 | Time Period | "Q2 2023" | Second quarter of 2023]

Relations:
[relation | E1 | received approval from | E2 | "received FDA approval"]
[relation | E2 | approved | E3 | "approval for Beovu®"]
[relation | E3 | approved in | E4 | "in Q2 2023"]

Text Input:
"{input_text}"

Output:
    """
    return prompt


def prompt_for_entity_extraction(input_text):
    prompt = f"""
--Objective--
Extract entities from the given text.

--Entity Extraction Rules--
1. Assign sequential EIDs starting from E1.
   - Maintain exact case from the text.
   - Merge coreferences before assigning EIDs.
2. Use specific, natural entity types (e.g., "Medical Device" not "PRODUCT").
   - Types can include a wide range of categories such as People, Organizations, Locations, Events, Time Periods, Products, Concepts, General Entities, Event Entities, etc.
3. Format entities as:
   [entity | EID | Type | "Entity Name" | Description]


--Consistency Checks--
1. EIDs must be sequential without gaps or duplicates.
   - EID creation only in the entity section.

--Processing Steps--
1. Extract all entities.
2. Resolve coreferences.

--Example--
Text Input:
"Novartis received FDA approval for Beovu® antibody in Q2 2023."

Output:
Entities:
[entity | E1 | Company | "Novartis" | Pharmaceutical corporation]
[entity | E2 | Regulatory Agency | "FDA" | US medical approval body]
[entity | E3 | Drug | "Beovu®" | Ophthalmic antibody treatment]
[entity | E4 | Time Period | "Q2 2023" | Second quarter of 2023]


Text Input:
"{input_text}"

Output:
    """
    return prompt



def prompt_for_score_triples(query, context, relevant_context):
    prompt = f"""

- Purpose -

Your goal is to assign **fine-grained relevance scores** to each triplet, in order to support **precise path reasoning and effective ranking**.  
Be **fair**, **consistent**, and **conservative with high scores** — only truly indispensable facts should receive the highest marks.



- Contextual Integration -

The background knowledge helps define the **reasoning context**.  
If a triplet connects **indirectly** to the question via a **chain formed with background triplets**, it may still be relevant and **scored ≥ 0.4**.



- Usage Note -

Your scoring will be used to **compare the utility of triplets in reasoning chains**.  
⚠️ **Be especially strict when assigning high scores (0.9 or 1.0)**. They indicate crucial reasoning components or answer-defining facts.



- Goal -

Score the contribution of each knowledge graph triplet (head, relation, tail) to the given question on a **seven-level scale from 0.0 to 1.0**.



- Instructions -

1. **Scoring Criteria**  
   Evaluate each triplet by considering both the `Question` and the `Background Knowledge`. Assign a score from this list:  
   **[1.0, 0.9, 0.7, 0.5, 0.3, 0.1, 0.0]**

   - **1.0 (Answer Core)**: Directly contributes a key part of the final answer. **The answer cannot be obtained without it**.

   - **0.9 (Answer Support)**: Adds essential specificity or disambiguation. **Important but potentially replaceable**.

   - **0.7 (Reasoning Bridge)**: Critical for connecting multiple pieces of knowledge, enabling correct reasoning.

   - **0.5 (Mildly Useful)**: Supports or enriches the reasoning process, but is **not required**.

   - **0.3 (Peripheral Context)**: Describes relevant entities but has **no impact** on reaching the answer.

   - **0.1 (Weak Association)**: Barely connected. Mentions entities in the question but offers **no useful content**.

   - **0.0 (Unrelated)**: Completely irrelevant to question logic, background, or entities.



2. **Background Knowledge Usage**:

   Consider how each triplet **relates to or complements the background knowledge**.  
   Triplets that help form a **reasoning path** with the background may merit a higher score.



3. **Output Format**:

   - Keep the **original order** of the input triplets.
   - Format each line as:

     `<head_entity | relation | tail_entity>: score`

   - Use **exactly one decimal place** for all scores.
   - Do **not** include explanations, markdown, or blank lines.



- Example -



Background Knowledge (Previous Path Triplets):

<Inception | released in | 2010>  
<Inception | genre | science fiction>



Question:

"Which author wrote the novel that was adapted into a 2015 sci-fi movie directed by Ridley Scott?"



Input Triplets:

<The Martian | released in | 2015>
<The Martian | directed by | Ridley Scott>
<The Martian | genre | science fiction>
<The Martian | based on | The Martian (novel)>
<The Martian (novel) | author | Andy Weir>
<Interstellar | released in | 2014>
<Interstellar | directed by | Christopher Nolan>
<Gravity | directed by | Alfonso Cuarón>
<Andy Weir | wrote | Artemis>
<Artemis | genre | science fiction>



Output:

<The Martian | released in | 2015>: 1.0  
<The Martian | directed by | Ridley Scott>: 1.0  
<The Martian | genre | science fiction>: 0.5  
<The Martian | based on | The Martian (novel)>: 1.0  
<The Martian (novel) | author | Andy Weir>: 1.0  
<Interstellar | released in | 2014>: 0.3  
<Interstellar | directed by | Christopher Nolan>: 0.3  
<Gravity | directed by | Alfonso Cuarón>: 0.3  
<Andy Weir | wrote | Artemis>: 0.1  
<Artemis | genre | science fiction>: 0.0


- Critical Rules -

⚠️ DO NOT:

- Add explanations or modify any triplet content.
- Reorder or omit any triplets.
- Use scores not in the list [1.0, 0.9, 0.7, 0.5, 0.3, 0.1, 0.0].



⚠️ MUST:

- Maintain **exact formatting** and **input order**.
- Output **exactly one line per triplet**, no empty lines.
- Use **one decimal place** for all scores.
- The output can not include triples in the background.

- Input Data -

Background Knowledge (Previous Path Triplets):

{relevant_context}



Question:

{query}



Input Triplets:

{context}



Output:

"""
    return prompt


def prompt_for_score_triples_old(query, context, relevant_context):
    prompt = f"""
- Goal -
Score the contribution of each knowledge graph triplet (head, relation, tail) to the given question on a scale of 0 to 1.

- Instructions -
1. **Scoring Criteria** (flexible ranges for broader generalization):  
   - **0.8–1.0**: Triplet directly answers the question or provides critical information for the answer.  
   - **0.4–0.7**: Triplet is indirectly related but could support the answer through inference or further exploration (e.g., linked entities, contextual details).  
   - **0.1–0.3**: Triplet has weak relevance but may contribute to contextual understanding or path expansion.  
   - **0.0**: Triplet is completely irrelevant to the question.  


2. **Output Format**:
   - Preserve the original triplet order
   - Format each line as: 
     `<head_entity | relation | tail_entity>: score`
   - Use one decimal place for scores (e.g., 0.2, 0.9)

- Example -

Background Knowledge (Previous Paths Triplets):
<Inception | released in | 2010>  
<Inception | genre | science fiction>  

Question:  
"Which director of a 2010 sci-fi movie also directed a film released in 2014?"  

Input Triplets:
<Inception | directed by | Christopher Nolan>
<Interstellar | directed by | Christopher Nolan>
<Interstellar | released in | 2014>
<Interstellar | genre | science fiction>
<The Dark Knight | directed by | Christopher Nolan>

Output:
<Inception | directed by | Christopher Nolan>: 1.0
<Interstellar | directed by | Christopher Nolan>: 0.6
<Interstellar | released in | 2014>: 0.9
<Interstellar | genre | science fiction>: 0.3
<The Dark Knight | directed by | Christopher Nolan>: 0.1

- Critical Rules -
⚠️ DO NOT:
- Add explanations or extra text
- Modify triplet order
- Create new triplets
- Use markdown formatting

⚠️ MUST:
- Maintain exact triplet formatting from input
- Strictly preserve the order of the triplets.
- Use consistent decimal precision
- Adhere strictly to 0-1 scale
- Ensure the number of output triplets is consistent with the number of "Input Triplets"

- Input Data -
Background Knowledge (Previous Path Triplets):
{relevant_context}

Question:
{query}

Input Triplets:
{context}

Output:
"""
    return prompt


# def prompt_for_score_triples(query, context, relevant_context):
#     prompt = f"""
# - Goal -
# Score the contribution of each knowledge graph triplet (head, relation, tail) to the given question on a scale of 0.0 to 1.0.
#
# - Instructions -
# 1. **Scoring Criteria** (Assign the single best-fitting score to each triplet. **Crucially, evaluate each triplet based on how it connects to both the `Question` and the `Background Knowledge`**):
#    - **1.0 (Critical Answer)**: The triplet directly provides a piece of the final answer or satisfies a core constraint of the question. The question cannot be answered without it.
#    - **0.7 (Essential Link)**: The triplet is a key part of the reasoning chain, connecting other critical pieces of information to form the answer. It is not the answer itself but is crucial for the reasoning process.
#    - **0.4 (Relevant Context)**: The triplet provides background information related to an entity in the question, but this information is not necessary to find the final answer.
#    - **0.1 (Weak Association)**: The triplet is related to an entity in the question but provides information that is almost irrelevant to answering the current query.
#    - **0.0 (Completely Irrelevant)**: The triplet is entirely unrelated to the entities and logic of the question.
#
# 2. **Output Format**:
#    - Preserve the original order of the triplets.
#    - Format each line as:
#      `<head_entity | relation | tail_entity>: score`
#    - Use one decimal place for scores (e.g., 0.4, 1.0).
#
# - Example -
#
# Background Knowledge (Previous Path Triplets):
# <Inception | released in | 2010>
# <Inception | genre | science fiction>
#
# Question:
# "Which director of a 2010 sci-fi movie also directed a film released in 2014?"
#
# Input Triplets:
# <Inception | directed by | Christopher Nolan>
# <Interstellar | directed by | Christopher Nolan>
# <Interstellar | released in | 2014>
# <Interstellar | genre | science fiction>
# <The Dark Knight | directed by | Christopher Nolan>
#
# Output:
# <Inception | directed by | Christopher Nolan>: 1.0
# <Interstellar | directed by | Christopher Nolan>: 0.7
# <Interstellar | released in | 2014>: 1.0
# <Interstellar | genre | science fiction>: 0.4
# <The Dark Knight | directed by | Christopher Nolan>: 0.1
#
# - Critical Rules -
# ⚠️ DO NOT:
# - Add any explanations or extra text.
# - Modify the order of the triplets.
# - Create new triplets.
# - Use any markdown formatting.
#
# ⚠️ MUST:
# - Maintain the exact triplet formatting from the input.
# - Strictly preserve the order of the triplets.
# - Use consistent one-decimal precision.
# - Adhere strictly to the 0.0-1.0 scoring scale.
# - Ensure the number of output triplets is identical to the number of "Input Triplets".
#
# - Input Data -
# Background Knowledge (Previous Path Triplets):
# {relevant_context}
#
# Question:
# {query}
#
# Input Triplets:
# {context}
#
# Output:
# """
#     return prompt

def prompt_for_eval_sufficiency(text, question):
    prompt = f"""
-Goal-
Given a question and the associated retrieved knowledge graph triplets (entity, relation, entity), you are asked to answer whether it's sufficient for you to answer the question with these triplets and your knowledge (Yes or No).

-Attentions-
 - Please strictly follow the format in the example to answer, do not provide additional content such as explanations.

-Example-

Contextual Information:

Triplets:
<Inception | released in | 2010> 
<Inception | genre | science fiction>
<Inception | directed by | Christopher Nolan>

Question:
"Who directed the movie Inception?"

Output:
YES

-Real Data-

Contextual Information:
{text}

Question:
{question}

Output:
    """
    return prompt


def prompt_for_missing_knowledge_identify(context_info, question):
    prompt = f"""
--Goal--
Given the provided contextual information and the question, analyze what specific information is missing—but necessary or helpful—to fully, accurately, and contextually answer the question.
Then, express each missing piece of information as an independent, atomic question.

--Instructions--
- Carefully review the Contextual Information (including Triples and Related Text) and ensure that you only generate questions for information not already explicitly provided, but still required or useful to answer the original question comprehensively.
- In addition to the core facts directly required by the question, also consider any relevant details, background, context, definitions, conditions, processes, or clarifications that would make the answer more complete or informative.
- Each generated question should address one clear, specific unknown, phrased as independently and atomically as possible.
- If multiple facts or aspects are needed for a thorough answer, break them down into the smallest possible units—each question should target only one fact, attribute, or relationship.
- Avoid redundancy; do not ask about information already present in the Contextual Information.
- Output only the missing questions; do not include any explanations or extra content.

--Example 1--

Contextual Information:
Triples:
<Albert Einstein | born in | Ulm>
<Albert Einstein | born on | 14 March 1879>
<Albert Einstein | known for | theory of relativity>
<Albert Einstein | won Nobel Prize in | 1921>
<1921 Nobel Prize in Physics | awarded to | Albert Einstein>

Question: "For what did Albert Einstein receive the Nobel Prize in Physics?"

Output:
What was the 1921 Nobel Prize in Physics awarded for?
What was the official reason or citation for awarding Albert Einstein the 1921 Nobel Prize in Physics?
Who were the other nominees for the 1921 Nobel Prize in Physics?
What were the criteria for awarding the Nobel Prize in Physics in 1921?

--Example 2--

Contextual Information:
Triples:
<The Moon | orbits | Earth>
<Earth | part of | Solar System>
<Solar System | includes | Sun>

Question: "How long does it take for the Moon to complete one orbit around the Earth, and how far is the Moon from the Earth?"

Output:
What is the orbital period of the Moon around the Earth?
What is the average distance from the Moon to the Earth?
What factors affect the Moon’s orbital period?
What is the shape of the Moon’s orbit around the Earth?

--Example 3--

Contextual Information:
Triples:
<Apple Inc. | founded by | Steve Jobs>
<Apple Inc. | founded on | April 1, 1976>
<Apple Inc. | headquarters in | Cupertino, California>

Question: "Where was Apple Inc. originally founded and who were its founders?"

Output:
Where was Apple Inc. originally founded?
Who, besides Steve Jobs, founded Apple Inc.? 
What was the original name of Apple Inc. at the time of founding?
What were the circumstances or motivations behind the founding of Apple Inc.?

--Real Data--

Contextual Information:
{context_info}

Question: "{question}"

Output:
"""
    return prompt

def prompt_for_missing_knowledge_extraction(context_info, sub_questions, context):
    prompt = f"""
--Goal--
Given the provided Contextual Information—including Existing Triples, Sub-questions, and Related Sentences—extract all new and relevant triples from the Related Sentences that (1) directly provide information necessary to answer the Sub-questions or add meaningful context, and (2) are not already present in the Existing Triples. The output must strictly follow the format and requirements below.

--Detailed Instructions--

Analyze the Sub-questions: Carefully read all Sub-questions to identify the specific information required, as well as any supporting or contextual details that would help answer them.
Locate Supporting Information: Extract information found directly in the “Related Sentences” that is relevant to the Sub-questions, including both direct answers and closely related facts or context.
Triple Extraction Criteria:
Extract any information that directly supports or adds relevant context to the Sub-questions.
Extracted relations must not already exist in the Existing Triples.
Each triple must have clear and explicit evidence from the Related Sentences (i.e., a supporting phrase from the original text).
Do not infer, expand, or merge information—extract only what is explicitly stated or unambiguously implied.
Ensure entity definitions are consistent and unambiguous (e.g., person names, locations, organizations).
Entities may be extracted multiple times if necessary (i.e., repeated entity extraction is allowed for clarity or completeness).
Relations must not be duplicated—each relation should be unique in the output.
If a sentence provides multiple relevant details, extract each as a separate triple.
Output Requirements:
Output only Entities and Relations, strictly following the format below.
Do not include explanations, comments, or any extra content.
Before outputting, ensure all new relations are not duplicated in the Existing Triples or within your output.
Every EntityID (EID) used in Relations must be present and defined in Entities.
In Relations, both SubjectEntityID and ObjectEntityID must be EIDs, not entity names.
--Output Format--
Entities:
[entity | EntityID | Type | "Name" | Description]

Relations:
[relation | SubjectEntityID | Predicate | ObjectEntityID | "Evidence or supporting phrase"]

--Example 1--
Existing Triples:
<Marie Curie | won Nobel Prize in Physics | 1903>
<Marie Curie | won Nobel Prize in Chemistry | 1911>

Sub-questions:
When did Marie Curie win her Nobel Prizes?
Who did Marie Curie share the 1903 Nobel Prize in Physics with?

Related Sentences:
Nobel Prizes | Marie Curie shared the 1903 Nobel Prize in Physics with Pierre Curie and Henri Becquerel.

Output:
Entities:
[entity | E1 | Person | "Marie Curie" | Physicist and chemist]
[entity | E2 | Person | "Pierre Curie" | Physicist]
[entity | E3 | Person | "Henri Becquerel" | Physicist]

Relations:
[relation | E1 | shared Nobel Prize with | E2 | "shared the 1903 Nobel Prize in Physics with Pierre Curie"]
[relation | E1 | shared Nobel Prize with | E3 | "shared the 1903 Nobel Prize in Physics with Henri Becquerel"]

--Example 2--
Existing Triples:
<World War II | ended in | 1945>
<World War II | involved | Germany>

Related Sentences:
Normandy Landings | The Normandy landings, also known as D-Day, occurred on June 6, 1944, and involved the United States, the United Kingdom, and Canada.

Sub-questions:
When did the Normandy landings take place?
Which countries were involved in the Normandy landings?

Output:
Entities:
[entity | E1 | Event | "Normandy landings" | Allied invasion of Normandy in World War II]
[entity | E2 | Country | "United States" | Country in North America]
[entity | E3 | Country | "United Kingdom" | Country in Europe]
[entity | E4 | Country | "Canada" | Country in North America]

Relations:
[relation | E1 | occurred on | E2 | "occurred on June 6, 1944"]
[relation | E1 | involved | E2 | "involved the United States"]
[relation | E1 | involved | E3 | "involved the United Kingdom"]
[relation | E1 | involved | E4 | "involved Canada"]

--Attention--

For every EntityID (EID) used in Relations, ensure the same EID is present and defined in Entities.
In Relations, both SubjectEntityID and ObjectEntityID must be E<IDs> format (not entity names).
--Real Data--

Existing Triples:
{context_info}

Sub-questions:
{sub_questions}

Related Sentences:
{context}

Output:
"""
    return prompt

def prompt_for_missing_knowledge_extraction_refine(context_info, sub_questions, context):
    prompt = f"""
--Goal--
Given the provided Contextual Information—including Existing Triples, Sub-questions, and Related Sentences—extract all new and relevant triples from the Related Sentences that (1) directly provide information necessary to answer the Sub-questions or add meaningful context, and (2) are not already present in the Existing Triples. The output must strictly follow the format and requirements below.

--Detailed Instructions--

1. Analyze the Sub-questions: Carefully read all Sub-questions to identify the specific information required, as well as any supporting or contextual details that would help answer them.

2. Locate Supporting Information: Extract information found directly in the “Related Sentences” that is relevant to the Sub-questions, including both direct answers and closely related facts or context.

3. Triple Extraction Criteria:
   - Extract any information that directly supports or adds relevant context to the Sub-questions.
   - Extracted relations must not already exist in the Existing Triples.
   - Each triple must have clear and explicit evidence from the Related Sentences (i.e., a supporting phrase from the original text).
   - Do not infer, expand, or merge information—extract only what is explicitly stated or unambiguously implied.
   - Ensure entity definitions are consistent and unambiguous (e.g., person names, locations, organizations).
   - Entities may be extracted multiple times if necessary (i.e., repeated entity extraction is allowed for clarity or completeness).
   - Relations must not be duplicated—each relation should be unique in the output.
   - If a sentence provides multiple relevant details, extract each as a separate triple.

--Entity Extraction Rules--
1. Entity Identification:
   - Assign sequential EIDs starting from E1 (E1, E2, E3, ...).
   - Maintain exact case from source text.
   - Merge references to the same entity before assigning EIDs, and prohibit the use of non-explicit pronouns such as he, she, the film, etc.
   - Format entities as:
     [entity | EID | Type | "Entity Name" | Description]

2. Entity Requirements:
   - Types can include a wide range of categories such as People, Organizations, Locations, Events, Time Periods, Products, Concepts, General Entities, Event Entities, etc.
   - Type must use specific natural categories (e.g., "Medical Device" not "PRODUCT").
   - Include functional context in description.

--Relationship Extraction Rules--
1. Validation Requirements:
   - Verify existence of both EIDs in entity list.
   - Prohibit relations with unregistered EIDs.
   - Require direct textual evidence.
   - In Relations, both SubjectEntityID and ObjectEntityID must be EIDs, not entity names.
   - Block relations where EID gap > current entity count.

2. Format Enforcement:
   - Relation format: [relation | SourceEID | RelationType | TargetEID | "exact quote"]
   - Each relation must be unique and not duplicated in the Existing Triples or within your output.

--Output Requirements--
- Output only Entities and Relations, strictly following the format below.
- Do not include explanations, comments, or any extra content.
- Before outputting, ensure all new relations are not duplicated in the Existing Triples or within your output.
- Every EntityID (EID) used in Relations must be present and defined in Entities.

--Output Format--
Entities:
[entity | EntityID | Type | "Name" | Description]

Relations:
[relation | SubjectEntityID | Predicate | ObjectEntityID | "Evidence or supporting phrase"]

--Example 1--
Existing Triples:
<Marie Curie | won Nobel Prize in Physics | 1903>
<Marie Curie | won Nobel Prize in Chemistry | 1911>

Sub-questions:
When did Marie Curie win her Nobel Prizes?
Who did Marie Curie share the 1903 Nobel Prize in Physics with?

Related Sentences:
Nobel Prizes | Marie Curie shared the 1903 Nobel Prize in Physics with Pierre Curie and Henri Becquerel.

Output:
Entities:
[entity | E1 | Person | "Marie Curie" | Physicist and chemist]
[entity | E2 | Person | "Pierre Curie" | Physicist]
[entity | E3 | Person | "Henri Becquerel" | Physicist]

Relations:
[relation | E1 | shared Nobel Prize with | E2 | "shared the 1903 Nobel Prize in Physics with Pierre Curie"]
[relation | E1 | shared Nobel Prize with | E3 | "shared the 1903 Nobel Prize in Physics with Henri Becquerel"]

--Example 2--
Existing Triples:
<World War II | ended in | 1945>
<World War II | involved | Germany>

Related Sentences:
Normandy Landings | The Normandy landings, also known as D-Day, occurred on June 6, 1944, and involved the United States, the United Kingdom, and Canada.

Sub-questions:
When did the Normandy landings take place?
Which countries were involved in the Normandy landings?

Output:
Entities:
[entity | E1 | Event | "Normandy landings" | Allied invasion of Normandy in World War II]
[entity | E2 | Country | "United States" | Country in North America]
[entity | E3 | Country | "United Kingdom" | Country in Europe]
[entity | E4 | Country | "Canada" | Country in North America]

Relations:
[relation | E1 | occurred on | E2 | "occurred on June 6, 1944"]
[relation | E1 | involved | E2 | "involved the United States"]
[relation | E1 | involved | E3 | "involved the United Kingdom"]
[relation | E1 | involved | E4 | "involved Canada"]

--Real Data--

Existing Triples:
{context_info}

Sub-questions:
{sub_questions}

Related Sentences:
{context}

Output:
"""
    return prompt

def prompt_for_focused_multi_relation_completion(context_info: str, entity_pairs: str, query: str) -> str:
    """
    生成一个精简的提示词，引导 LLM 优先抽取与一个问题 (query) 相关的多个关系。

    此版本在保持原有核心指令不变的情况下，尽可能缩短了提示词的长度。

    :param context_info: 上下文信息。
    :param entity_pairs: 格式化后的实体对字符串，每对占一行。
    :param query: 一个用作抽取焦点的问题。
    :return: 完整的、精简后的提示词字符串。
    """
    prompt = f"""
-Rules-
For each entity pair, extract all relationships from the Context.
1.  **Prioritize**: First, find relationships that answer the "Focus Question". Second, find any other relationships.
2.  **Format**: List all relationships for a pair on one line, separated by "|". It's good practice to list question-relevant ones first.
3.  **No Relation**: If no relationship is found, you MUST output "none".
4.  **Strictness**: You MUST generate exactly one output line for each entity pair. Do not add any explanations.
5.  The complete entity pair must be outputted regardless,even if the relationship is none,for example, "Entity 1 || Entity 2: none".
-Example-
Focus Question:
What were the corporate leadership roles?

Context:
As CEO and co-founder of Apple, Steve Jobs drove the creation of the iPhone. Apple, the company he co-founded, is the developer of the iPhone.

Entity Pairs:
Steve Jobs || Apple
Apple || iPhone
Steve Jobs || iPhone

Output:
Steve Jobs || Apple: CEO | co-founder
Apple || iPhone: developer_of
Steve Jobs || iPhone: none

-Task-
Focus Question:
{query}

Context:
{context_info}

Entity Pairs:
{entity_pairs}

Output:
"""
    return prompt

def prompt_for_summary_answer(explanatory_answer, question):
    prompt = f"""
# Role
You are an information extraction expert.

# Task
Your task is to extract a precise and concise answer for the given `Question` based *strictly* on the provided `Explanatory Answer`.

# Core Rules
1.  **Extract, Don't Infer**: Your answer must be found directly within the text of the `Explanatory Answer`. Do not infer, guess, or use any external knowledge.
2.  **Be Concise**: Output only the final answer. The answer should be as short as possible (e.g., "Barack Obama", "2016", "yes"). Do not add any explanations, labels, or introductory phrases like "The answer is:".
3.  **Handle Unknowns**: If the answer cannot be found in the `Explanatory Answer`, you must output the single word: `unknown`.

# Examples
Explanatory Answer:
Barack Obama served as the 44th President of the United States from 2009 to 2017.

Question:
"Who was the 44th President of the United States?"
Output:
Barack Obama

Explanatory Answer:
The capital of France is Paris.

Question:
"What is the capital of France?"
Output:
Paris

Explanatory Answer:
There is no information about the president in the text.

Question:
"Who is the president?"
Output:
unknown

# Execute
Explanatory Answer:
###########
{explanatory_answer}
###########
Question:
"{question}"
Output:
"""
    return prompt

def prompt_for_llm_only(question):
    prompt = f"""
-Goal-
 Given a question, use your own knowledge base to answer the question.
 You should provide an answer with a high level of credibility.
 

Question:
{question}
Output:
"""
    return prompt

def prompt_for_llm_only_(question):
    prompt = f"""
-Goal-
 Given a question, use your own knowledge base to answer the question.

-Attentions-
 - If you are not certain of the answer, or the answer is not in your knowledge base, reply only with "unknown".
 - Please strictly follow the format in the example to answer, do not provide additional content such as explanations.
 - The answer needs to be as precise and concise as possible such as "Flatbush section of Brooklyn, New York City", "Christopher Nolan", "New York City".
 - Ensure that the answer corresponds exactly to the Question without deviation.

-Example-
Question:
"Who directed the movie Inception?"
Output:
Christopher Nolan

Question:
"Who founded the Appple?"
Output:
Steve Jobs

Question:
"What is the capital of Atlantis?"
Output:
unknown

-Real Data-
Question:
{question}
Output:
"""
    return prompt

def prompt_for_RAG_llm(context, question):
    prompt = f"""
-Goal-
Given the question, use the text information I provided to answer the question.
You should answer strictly based on the information provided in the context.

Text Information:
{context}

Question:
"{question}"
Output:
"""
    return prompt

def prompt_for_query_reasoning_path_generate(question, K):

    case = """
{
  "type_chains": [
    {
      "path": [
        {"type":"<Type 1>"},
        {"rel":"<Relation A>"},
        {"type":"<Type 2>"},
        {"rel":"<Relation B>"},
        {"type":"<Type 3>"}
      ],
      "rationale": "Brief explanation of why this path is plausible in this question's context"
    }
    // At least {K} paths
  ]
}
    """
    prompt = f"""
You are an Entity Type Reasoning Path Generator.  
Given a natural language question, output multiple possible Entity Type Reasoning Paths.  

Requirements:
1. Do NOT use any predefined entity types. Instead, infer appropriate, domain-specific type names from the question itself.  
   - Avoid generic labels such as "Organization" or "Person"; use contextualized names like "BoardSeatGrantor", "EquityStakeTarget", etc.
2. Each path should consist of alternating [Entity Type] and (Relation Type) elements, connected with "->".
3. Output at least {K} different paths.  
   - Cover variations such as:
     - Synonymous or reversed relations  
     - Entity granularity shifts (instance ↔ type)  
     - Possible intermediate layers (e.g., platform, protocol, subsidiary, acquisition)
4. For each path, include a short rationale explaining why it makes sense in the context of the question.
5. Output valid JSON for easy parsing.

[Input Question]
{question}

[Output Format Example]
{case}
"""
    return prompt

def prompt_for_coref_disambiguation(title: str, prev_chunk: str, current_chunk: str) -> str:
    return f"""

--GOAL--
Rewrite the CURRENT TEXT CHUNK into a self-contained version where all pronouns, abbreviations, and ambiguous references are replaced with explicit and fully understandable expressions. The rewritten text should be interpretable on its own, without relying on context.

--REQUIREMENTS--
- If a pronoun or ambiguous reference in the current chunk can be clearly linked to something mentioned in the previous disambiguated text, replace it with the full explicit expression.
- If an abbreviation can be resolved based on the previous disambiguated text, use the full form or “Full Form (Abbreviation)”.
- Do not add any new information that does not appear in either the previous disambiguated text or the current chunk.
- Preserve the original meaning and tone.
- Output should should be included within <TEXT></TEXT>.

--EXAMPLE--
INPUT:
[Document Title]
The Development of General Relativity

[Previous Disambiguated Text]
Albert Einstein proposed General Relativity in 1915.

[Current Text Chunk]
Later, he published it in a journal.

OUTPUT:
<TEXT>
Albert Einstein later published General Relativity in a journal.
</TEXT>

--REAL DATA--
INPUT:
[Document Title]
{title}

[Previous Disambiguated Text]
{prev_chunk}

[Current Text Chunk]
{current_chunk}

OUTPUT:
"""


def prompt_for_KG_entities(input_text: str) -> str:
    """
    Optimized prompt: extracts ONLY entities and returns them wrapped in <ENTITY>...</ENTITY>.
    """
    prompt = f"""
--Objective--
Extract all **entities only** from the given text, assign strict sequential IDs (E1, E2, ...), and return the list **wrapped inside a single <ENTITY>...</ENTITY> block**.

--Hard Output Contract (must follow exactly)--
- Output **one** block:
  <ENTITY>
  [entity | E1 | Type | "Entity Name" | Description]
  [entity | E2 | Type | "Entity Name" | Description]
  ...
  </ENTITY>
- If no entities are found, output:
  <ENTITY></ENTITY>

--Entity Extraction Rules--
1) Identification & IDs
   - Assign continuous EIDs starting from E1 with no gaps or duplicates.
   - One line per entity, format EXACTLY:
     [entity | EID | Type | "Entity Name" | Description]
   - Preserve the exact surface form/casing for "Entity Name" as in the text (verbatim span).
   - Merge coreferent mentions BEFORE assigning EIDs (resolve pronouns like he/she/it/the film/etc. to explicit entities when unambiguous; otherwise skip).

2) Typing
   - Use specific, natural categories (e.g., "Pharmaceutical Company", "Regulatory Agency", "Drug", "Medical Device", "Research Paper", "Conference", "Standard", "Time Period", "Geopolitical Location", "Product", "Concept", "Event", "Organization", "Person").
   - Prefer the most concrete type supported by the text; avoid generic placeholders like "THING" or "PRODUCT" unless unavoidable.

3) Descriptions
   - Provide a concise role/function grounded in the text.
   - Must NOT introduce facts not supported by the text.

--Consistency & Safety Checks (must satisfy)--
- EIDs: continuous from E1; no missing indices; unique.
- No relations, attributes, or extra sections.
- Only lines starting with "[entity |" inside <ENTITY>...</ENTITY>.

--Example--
TEXT:
"GraphRAG was introduced by Microsoft researchers in 2023. 
It enhances large language models by retrieving knowledge from structured graphs. 
The technique was demonstrated using the Wikidata knowledge graph."

OUTPUT:
<ENTITY>
[entity | E1 | Organization | "Microsoft" | Research and technology company]
[entity | E2 | Method | "GraphRAG" | Retrieval-augmented graph-based reasoning approach]
[entity | E3 | Time Period | "2023" | Year of introduction]
[entity | E4 | Knowledge Graph | "Wikidata" | Collaborative structured knowledge base]
</ENTITY>

--Input--
TEXT:
"{input_text}"

OUTPUT:
"""
    return prompt


def prompt_for_extract_entities(input_text: str) -> str:
    """
    Optimized prompt: extracts ONLY entities and returns them wrapped in <ENTITY>...</ENTITY>.
    """
    prompt = f"""
--Objective--
Extract all **entities only** from the given text, assign strict sequential IDs (E1, E2, ...), and return the list **wrapped inside a single <ENTITY>...</ENTITY> block**.

--Hard Output Contract (must follow exactly)--
- Output **one** block:
  <ENTITY>
  [entity | E1 | Type | "Entity Name" | Description]
  [entity | E2 | Type | "Entity Name" | Description]
  ...
  </ENTITY>
- If no entities are found, output:
  <ENTITY></ENTITY>

--Entity Extraction Rules--
1) Identification & IDs
   - Assign continuous EIDs starting from E1 with no gaps or duplicates.
   - One line per entity, format EXACTLY:
     [entity | EID | Type | "Entity Name" | Description]
   - Preserve the exact surface form/casing for "Entity Name" as in the text (verbatim span).
   - Merge coreferent mentions BEFORE assigning EIDs (resolve pronouns like he/she/it/the film/etc. to explicit entities when unambiguous; otherwise skip).

2) Typing
   - Use specific, natural categories (e.g., "Pharmaceutical Company", "Regulatory Agency", "Drug", "Medical Device", "Research Paper", "Conference", "Standard", "Time Period", "Geopolitical Location", "Product", "Concept", "Event", "Organization", "Person").
   - Prefer the most concrete type supported by the text; avoid generic placeholders like "THING" or "PRODUCT" unless unavoidable.

3) Descriptions
   - Provide a concise role/function grounded in the text.
   - Must NOT introduce facts not supported by the text.

--Consistency & Safety Checks (must satisfy)--
- EIDs: continuous from E1; no missing indices; unique.
- No relations, attributes, or extra sections.
- Only lines starting with "[entity |" inside <ENTITY>...</ENTITY>.

--Example--
TEXT:
"GraphRAG was introduced by Microsoft researchers in 2023. 
It enhances large language models by retrieving knowledge from structured graphs. 
The technique was demonstrated using the Wikidata knowledge graph."

OUTPUT:
<ENTITY>
[entity | E1 | Organization | "Microsoft" | Research and technology company]
[entity | E2 | Method | "GraphRAG" | Retrieval-augmented graph-based reasoning approach]
[entity | E3 | Time Period | "2023" | Year of introduction]
[entity | E4 | Knowledge Graph | "Wikidata" | Collaborative structured knowledge base]
</ENTITY>

--Input--
TEXT:
"{input_text}"

OUTPUT:
"""
    return prompt


def prompt_for_extract_relations(input_text: str, entities_block: str) -> str:
    """
    Optimized prompt for extracting relations strictly among pre-identified entities.
    Output must be wrapped inside <RELATION>...</RELATION>.
    """
    prompt = f"""
--Objective--
Extract ONLY explicit and verifiable relations **between the pre-extracted entities below**.
Do NOT invent new entities, relations, or inferred links.

--Output Contract--
Return exactly one block:
<RELATION>
[relation | SourceEID | RelationType | TargetEID | "exact quote"]
...
</RELATION>
- No headers, no comments, no code fences.
- Output <RELATION></RELATION> if no valid relations.

--Rules--
1. EID Validation
   - Use only EIDs listed below (case-sensitive, continuous).
   - Any missing or malformed EID invalidates the relation.

2. Evidence
   - Each relation must be supported by an exact text span (verbatim substring).
   - The quoted evidence must appear literally in the input text.

3. RelationType
   - Concise and semantically grounded (e.g., “approved”, “located in”, “founded by”).
   - Reflects the direction implied by text (Source → Target).

4. Precision & Rejection
   - Omit vague patterns like “related to”, “associated with”.
   - Reject speculative or inferred connections.
   - No paraphrasing or summarization.

--Example--
TEXT:
"Novartis received FDA approval for Beovu® in Q2 2023."

Entities:
[entity | E1 | Company | "Novartis" | Pharmaceutical corporation]
[entity | E2 | Regulatory Agency | "FDA" | US medical approval body]
[entity | E3 | Drug | "Beovu®" | Ophthalmic antibody]
[entity | E4 | Time | "Q2 2023" | Second quarter of 2023]

OUTPUT:
<RELATION>
[relation | E1 | received approval from | E2 | "received FDA approval"]
[relation | E2 | approved | E3 | "approval for Beovu®"]
[relation | E3 | approved in | E4 | "in Q2 2023"]
</RELATION>

--Input--
TEXT:
"{input_text}"

Entities:
{entities_block}

OUTPUT:
"""
    return prompt


def prompt_for_preprocess_query(input_text: str) -> str:
    """
    Build a strict instruction prompt for extracting entities/keywords
    directly mentioned in a user query.

    The model MUST:
    - Always return at least one entity line.
    - Wrap the final output with <ENTITY> ... </ENTITY>.
    """
    prompt = f"""
--Objective--
Analyze the logical structure of the query to determine the number of information retrieval paths required.
Extract only the "Main Topic Entities" necessary to traverse these paths.
Only extract strings that explicitly appear in the query text.

--Path Analysis & Selection Logic--
1. **Single Path (1 Entity):**
   - If the query asks for attributes, facts, or relations regarding a SINGLE subject.
   - Example: "Who is the director of [Self-Made Maids]?"
   - Action: Extract exactly ONE entity (the main subject).

2. **Dual Path (2 Entities):**
   - If the query requires comparing, differentiating, or connecting TWO distinct subjects.
   - Example: "Which director was born first, [Self-Made Maids] or [A Day For Lionhearts]?"
   - Action: Extract exactly TWO entities (the two subjects being compared/connected).

--Entity Extraction Rules--
1) Assign sequential EIDs starting from E1.
2) Preserve the exact surface form from the query.
3) **Strict Filtering:** Ignore auxiliary entities (dates, locations, generic nouns) UNLESS they are one of the Main Topic Entities identified in the Path Analysis.
4) Use specific, natural types (e.g., "Movie", "Person", "Book", "Organization", etc.).
5) Each entity line MUST follow exactly:
   [entity | EID | Type | "Entity Name" | Brief Description (<=10 words)]
6) If the query logic is ambiguous or lacks specific entities, treat the whole query as E1 (Concept).

--Output Format (STRICT)--
Return ONLY the following block, with no prose before or after:
<ENTITY>
[entity | E1 | Type | "..." | ...]
[entity | E2 | Type | "..." | ...]
...
</ENTITY>

--Examples--

Input: "What is the population of Tokyo?"
Logic: Single Path (Fact about Tokyo).
Output:
<ENTITY>
[entity | E1 | Location | "Tokyo" | Target city]
</ENTITY>

Input: "Which film whose director was born first, Self-Made Maids or A Day For Lionhearts?"
Logic: Dual Path (Requires looking up Director of Movie A AND Director of Movie B, then comparing).
Output:
<ENTITY>
[entity | E1 | Work | "Self-Made Maids" | Film A for comparison]
[entity | E2 | Work | "A Day For Lionhearts" | Film B for comparison]
</ENTITY>

Input: "Who led the project in Europe?"
Logic: Fallback (No specific entity).
Output:
<ENTITY>
[entity | E1 | Concept | "Who led the project in Europe?" | Full query as fallback]
</ENTITY>

--Current Input--
Text Input:
"{input_text}"

Output:
"""
    return prompt


def prompt_for_preprocess_query_more_ent(input_text: str) -> str:
    """
    Build a strict instruction prompt for extracting entities/keywords
    directly mentioned in a user query.

    The model MUST:
    - Always return at least one entity line.
    - Wrap the final output with <ENTITY> ... </ENTITY>.
    """
    prompt = f"""
--Objective--
Extract one or more directly mentioned keywords or entities from the given query.
Only extract strings that explicitly appear in the query text. Do NOT invent, infer,
or normalize beyond what is written.

If the query contains no clearly identifiable entities by the rules below,
treat the entire original query string as ONE Concept entity and output it as E1.
The output MUST NEVER be empty.

--Entity Extraction Rules--
1) Assign sequential EIDs starting from E1 with no gaps or duplicates.
2) Preserve the exact surface form (case, spacing, punctuation) from the query
   for the "Entity Name".
3) Merge coreferences and keep only the main mention actually present in the text.
4) Use specific, natural types (e.g., "Person", "Location", "Event", "Organization",
   "Work", "Time", "Concept", "Metric", "Product", etc.).
5) Each entity line MUST follow exactly:
   [entity | EID | Type | "Entity Name" | Brief Description (<=10 words)]
6) If no extractable entities appear under the above rules, output exactly:
   [entity | E1 | Concept | "<full original query>" | "Full query as fallback"]

--Consistency & Safety Checks--
- EIDs must be continuous and unique.
- Do NOT add anything that is not literally present in the query text, except
  the fallback in Rule 6.
- The output MUST contain at least one entity line between <ENTITY> and </ENTITY>.

--Output Format (STRICT)--
Return ONLY the following block, with no prose before or after:
<ENTITY>
[entity | E1 | Type | "..." | ...]
[entity | E2 | Type | "..." | ...]
...
</ENTITY>

--Positive Example--
Text Input:
"What is the population of Tokyo in 2020?"

Output:
<ENTITY>
[entity | E1 | Location | "Tokyo" | Major city in Japan]
[entity | E2 | Time | "2020" | Calendar year]
</ENTITY>

--Fallback Example (no obvious entity, use full query)--
Text Input:
"Who led the project in Europe?"

Output:
<ENTITY>
[entity | E1 | Concept | "Who led the project in Europe?" | Full query as fallback]
</ENTITY>

Text Input:
"{input_text}"

Output:
"""
    return prompt


def prompt_for_reasoning(question, text):
    prompt = f"""
--Goal--
Answer the question strictly based on the **provided context**. If the context lacks required facts, use **internal knowledge**, but every such usage must be explicitly marked as (internal). Every statement must be traceable to either the context or internal knowledge.

--Reasoning Requirements--
You must produce a detailed reasoning section wrapped inside **<Thought>...</Thought>** that shows a rich, step-by-step thought process.

1. Overall Structure:
   - Inside <Thought>, present reasoning as a sequence of **numbered steps**: "1.", "2.", "3.", ...
   - Each step must have a **short bold title** describing its purpose.
   - Each step can contain multiple bullet points for sub-details, comparisons, or intermediate checks.

2. Logical Coverage:
   Your reasoning should be **rich and comprehensive**, typically covering (when applicable):

   - Identifying the key entities, time, place, or quantities involved.
   - Collecting all context sentences relevant to the question.
   - Comparing or reconciling potentially conflicting pieces of information.
   - Handling missing information (explicitly noting what is not in the context).
   - Deciding when internal knowledge is required and stating exactly what is retrieved.
   - Performing any needed calculations, conversions, or logical inferences.
   - Verifying consistency of the final answer with all used evidence.

   Do not skip directly from reading the context to the final answer; show intermediate thinking.

3. Evidence Attribution:
   - Every factual statement or extracted detail must end with:
       - **(context)** — when the information is taken directly from the provided text
       - **(internal)** — when the information comes from your own background knowledge
   - If a sentence combines multiple facts from different sources, split it into multiple sentences so that each has a single source tag.

4. Reasoning Style:
   - Be analytic and precise, but not verbose for no reason.
   - No storytelling, no rhetorical questions.
   - Every step must help move from the context to the final answer.

--Final Answer Requirements--
After the </Thought> block, output a section titled **Final Answer:** containing:

- One single short factual answer.
- No explanation or justification.
- Wrapped in **<ANSWER>...</ANSWER>**.
- Must always provide a concrete answer (never respond with “unknown” or similar).

--Rules--
- Do not restate or paraphrase the question.
- Do not output sections other than <Thought> and **Final Answer**.
- Do not add disclaimers or meta commentary.

--Example--
Context:
"Lana Wachowski directed The Matrix."

Question:
"When was Lana Wachowski born?"

Output:
<Thought>
1. **Identify target entity**:
   - The question asks about Lana Wachowski. (context)

2. **Scan context for birth information**:
   - The provided sentence mentions only that she directed The Matrix. (context)
   - There is no explicit birth date in the given text. (context)

3. **Decide whether internal knowledge is needed**:
   - The question requires a specific birth date, which is not present in the context. (context)
   - Therefore, I must use internal knowledge to answer the question. (internal)

4. **Retrieve birth date using internal knowledge**:
   - From internal knowledge, Lana Wachowski was born on June 21, 1965. (internal)

5. **Validate answer consistency**:
   - The birth date does not conflict with any information in the given context. (context)

6. **Prepare final answer**:
   - I can now provide the birth date as the final answer. (internal)
</Thought>
Final Answer:
<ANSWER>June 21, 1965</ANSWER>

--Real Data--
Context:
{text}

Question:
{question}

Output:
"""
    return prompt

def prompt_for_reasoning_simple(question, text):
    prompt = f"""
Answer the Question based on the Context.

STRICT FORMAT:
<Thought>
1. **Step Title**:
   - Reasoning details...
2. **Step Title**:
   - Reasoning details...
</Thought>
Final Answer:
<ANSWER>Concise Answer</ANSWER>

RULES:
1. Reasoning MUST be wrapped in <Thought>.
2. Use numbered steps with **Bold Titles**.
3. Keep logic concise and clear.

EXAMPLE:
Q: Who directed The Matrix?
C: The film The Matrix was directed by Lana Wachowski in 1999.
<Thought>
1. **Identify Subject**:
   - The question asks for the director of The Matrix.
2. **Extract Information**:
   - The context explicitly states Lana Wachowski directed it.
</Thought>
Final Answer:
<ANSWER>Lana Wachowski</ANSWER>

DATA:
Context: {text}
Question: {question}
"""
    return prompt
def prompt_for_evidence_entities(question: str, context: str) -> str:
    """
    Prompt template: extract evidence entities for QA and score their importance.
    Output format: all lines wrapped in a single <ENTITY>...</ENTITY> block,
    one entity per line as 'Entity Name: score'.
    """
    prompt = f"""
Task
----
You are an information extraction component. Your job is to identify and score entities that are useful as EVIDENCE for answering a QUESTION, given some CONTEXT passages.

You MUST NOT answer the question. You MUST ONLY output entities and their importance scores.

Input
-----
You receive:
- QUESTION: a natural language question.
- CONTEXT: one or more passages.

Read the QUESTION first, then the CONTEXT, and find entities that help answer the QUESTION.

Output (Hard Contract)
----------------------
You MUST follow these rules exactly:

1. Wrapper
   - Output EXACTLY ONE block:
     <ENTITY>
     ...
     </ENTITY>

2. Inside the block
   - Each non-empty line inside <ENTITY>...</ENTITY> represents exactly ONE entity.
   - Line format MUST be:

     Entity Name: score

   - Exactly one space before and after the colon.
   - Do NOT wrap entity names in quotes.
   - Do NOT output headings, explanations, JSON, XML, or any other text.
   - If no useful entities are found, output:
     <ENTITY></ENTITY>

3. Score
   - score ∈ [0.0, 1.0], inclusive.
   - Use exactly one decimal place (e.g., 1.0, 0.9, 0.5).

Scoring Meaning
---------------
- 1.0      : Core entity, almost indispensable to identify or express the answer.
- 0.7–0.9  : Very important; strongly narrows down or confirms the answer.
- 0.4–0.6  : Helpful but not core; supporting evidence or background.
- 0.1–0.3  : Weakly related; only marginally helpful.
- 0.0      : Usually not output; if this low, omit the entity.

Entity Selection Rules
----------------------
1) What counts as an entity
   - Persons, organizations, locations, events.
   - Works (papers, books, movies), methods, concepts.
   - Roles/titles, and time expressions when they help locate or support the answer.

2) How to select
   - From the QUESTION, infer what kind of answer is expected (who/what/where/etc.).
   - From the CONTEXT, pick entities that:
     - directly answer the question, OR
     - clearly identify the answer (e.g., full name, key concept), OR
     - provide strong supporting evidence for the answer.
   - Ignore entities that are clearly irrelevant to the QUESTION.

3) Naming
   - Use the surface form as it appears in QUESTION or CONTEXT.
   - Preserve original casing/spelling.
   - If different mentions (e.g., name vs. role) are each useful, you may output both.

4) Quantity
   - Do NOT try to list every possible entity.
   - Prefer a small, high-quality set of the most useful entities.

Illustrative Example
--------------------
QUESTION:
Which scientist proposed the theory of general relativity?

CONTEXT:
Albert Einstein
Albert Einstein was a German-born theoretical physicist who developed the theory of relativity, one of the two pillars of modern physics. His theory of general relativity, published in 1915, revolutionized our understanding of gravity.

Relativity
Relativity generally refers to two related theories by Albert Einstein: special relativity and general relativity.

VALID OUTPUT EXAMPLE:
<ENTITY>
Albert Einstein: 1.0
theory of general relativity: 0.9
theoretical physicist: 0.6
1915: 0.4
relativity: 0.3
</ENTITY>

Actual Input
------------
QUESTION:
{question}

CONTEXT:
{context}

Final Instruction
-----------------
Now, based on the QUESTION and CONTEXT above, output EXACTLY ONE block:

<ENTITY>
Entity Name: score
...
</ENTITY>

Do NOT output anything else.
"""
    return prompt

def prompt_for_evidence_entities_simple(question: str, context: str) -> str:
    """
    Simplified prompt: extract and score evidence entities.
    """
    prompt = f"""
--Task--
Identify entities in the CONTEXT that serve as evidence to answer the QUESTION.
Assign an importance score (0.0 to 1.0) to each entity.

--Output Format--
Return ONLY a single XML block. NO prose, NO markdown.
<ENTITY>
Entity Name: score
...
</ENTITY>

--Constraints--
1. **Format**: One entity per line as `Name: score` (e.g., `Apple Inc: 1.0`).
2. **Scoring**:
   - 1.0: Core entity (Direct Answer/Crucial).
   - 0.5-0.9: Important supporting evidence.
   - 0.1-0.4: Weakly related/Background.
3. **Extraction**: Use exact surface forms from the text.
4. **Content**: Do NOT answer the question. Only list evidence entities.

--Example--
Q: Who proposed general relativity?
C: Albert Einstein published the theory of general relativity in 1915.
<ENTITY>
Albert Einstein: 1.0
theory of general relativity: 0.9
1915: 0.5
</ENTITY>

--Input--
QUESTION:
{question}

CONTEXT:
{context}
"""
    return prompt

def prompt_for_read_and_reasoning(question: str, context: str) -> str:
    """
    Prompt template: perform passage-based reasoning and produce a structured Thought + Answer output.
    This template matches the QA pipeline design (system requires "Thought:" then "Answer:").
    """
    prompt = f"""
CONTEXT:
{context}

QUESTION:
{question}

Thought: 
"""
    return prompt
