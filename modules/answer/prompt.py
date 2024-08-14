from typing import List


def relation_prune_prompt(question: str, entity: str, relations: List[str], width: int):
    relations = '[' + ', '.join(relations) + ']'
    prompts = [
        dict(role='system', content="""You are an AI assistant tasked with helping users rank relations from a knowledge graph based on their relevance to a specific question. You should not fabricate new relations."""),
        dict(role='user', content=f"""Please retrieve the top {width} relations in the given list that contribute the most to answering the following question.""" + """Rate their contribution on a scale from 0.0 to 1.0. Ensure the sum of scores equals 1.0. Each relation and its score should be put in one line and formatted like {relation: <relation_name>, score: <score>}.\nQ: Name the president of the country whose main spoken language was Brahui in 1980?\nTopic Entity: Brahui Language\nRelations: [language.human_language.main_country, language.human_language.language_family, language.human_language.iso_639_3_code, base.rosetta.languoid.parent, language.human_language.writing_system, base.rosetta.languoid.languoid_class, language.human_language.countries_spoken_in, kg.object_profile.prominent_type, base.rosetta.languoid.document, base.ontologies.ontology_instance.equivalent_instances, base.rosetta.languoid.local_name, language.human_language.region]\nA: """),
        dict(role='assistant', content="""1. {relation: "language.human_language.main_country", score: 0.5}: This relation is highly relevant as it directly relates to the country whose president is being asked for, and the main country where Brahui language was spoken in 1980.\n2. {relation: "language.human_language.countries_spoken_in", score: 0.3}: This relation also provides information on the countries where Brahui language was spoken, which could help narrow down the search for the president.\n3. {relation: "base.rosetta.languoid.parent", score: 0.2}: This relation is less relevant but still provides some context on the language family to which Brahui belongs, which could be useful in understanding the linguistic and cultural background of the country in question."""),
        dict(role='user', content=f"""Q: {question}\nTopic Entity: {entity}\nRelations: [{relations}]\nA: """)
    ]
    return prompts


def entity_score_prompt(question: str, relation: str, entities: List[str]):
    entities = '[' + ', '.join(entities) + ']'
    prompts = [
        dict(role='system', content="""You are an AI assistant that helps people rank entities from a knowledge graph based on their relevance to specific questions."""),
        dict(role='user', content="""Please rate the entities in the given list that contribute the most to answering the following question with the specified relation, on a scale from 0.0 to 1.0. Ensure the sum of scores equals 1.0. Each entity and its score should be put in one line and formatted like {entity: <entity_name>, score: <score>}.\nQ: The movie featured Miley Cyrus and was produced by Tobin Armbrust?\nRelation: film.producer.film\nEntities: The Resident; So Undercover; Let Me In; Begin Again; The Quiet Ones; A Walk Among the Tombstones\nScore: """),
        dict(role='assistant', content="""{entity: "The Resident", score: 0.0}\n{entity: "So Undercover", score: 1.0}\n{entity: "Let Me In", score: 0.0}\n{entity: "Begin Again", score: 0.0}\n{entity: "The Quiet Ones", score: 0.0}\n{entity: "A Walk Among the Tombstones", score: 0.0}\n\nExplanation: The movie that matches the given criteria is "So Undercover" with Miley Cyrus and produced by Tobin Armbrust. Therefore, the score for "So Undercover" would be 1.0, and the scores for all other entities would be 0.0 because they do not meet both conditions of the question."""),
        dict(role='user', content=f"""Q: {question}\nRelation: {relation}\nEntities: {entities}\nScore: """)
    ]
    return prompts


def reasoning_prompt(question: str, triple_chains: List[List[List[str]]]):
    """
    :param question: QA question
    :type question: str
    :param triple_chains: shape = (chain, triple, element)
    :type triple_chains: list
    """
    flatten_chains = '\n'.join(['\n'.join([', '.join([e for e in triple]) for triple in triples]) for triples in triple_chains])
    prompts = [
        dict(role='system', content="""You are an AI assistant that helps people find information and answer questions using available knowledge triplets."""),
        dict(role='user', content="""Given a question and the associated retrieved knowledge triplets (entity, relation, entity), determine whether the information provided is sufficient to answer the question with {sufficient = True} or {sufficient = False} and explain your reasoning. Then, if your answer is {sufficient = True}, please output the answer by {answer = <your answer>}. Otherwise please output {answer = None}.\nQ: Find the person who said "Taste cannot be controlled by law", what did this person die from?\nKnowledge Triplets: Taste cannot be controlled by law., media_common.quotation.author, Thomas Jefferson\nA: """),
        dict(role='assistant', content="""{sufficient = False}. Based on the given knowledge triplets, it's not sufficient to answer the entire question. The triplets only provide information about the person who said "Taste cannot be controlled by law" which is Thomas Jefferson. To answer the second part of the question, it's necessary to have additional knowledge about where Thomas Jefferson's dead. {answer = None}"""),
        dict(role='user', content="""Q: The artist nominated for The Long Winter lived where?\nKnowledge Triplets: The Long Winter, book.written_work.author, Laura Ingalls Wilder\nLaura Ingalls Wilder, people.person.places_lived, Unknown-Entity\nUnknown-Entity, people.place_lived.location, De Smet\nA: """),
        dict(role='assistant', content="""{sufficient = True}. Based on the given knowledge triplets, the author of The Long Winter, Laura Ingalls Wilder, lived in De Smet. Therefore, the answer to the question is {answer = "De Smet"}."""),
        dict(role='user', content="""Q: Who is the coach of the team owned by Steve Bisciotti?\nKnowledge Triplets: Steve Bisciotti, sports.professional_sports_team.owner_s, Baltimore Ravens\nSteve Bisciotti, sports.sports_team_owner.teams_owned, Baltimore Ravens\nSteve Bisciotti, organization.organization_founder.organizations_founded, Allegis Group\nA: """),
        dict(role='assistant', content="""{sufficient = False}. Based on the given knowledge triplets, the coach of the team owned by Steve Bisciotti is not explicitly mentioned. However, it can be inferred that the team owned by Steve Bisciotti is the Baltimore Ravens, a professional sports team. Therefore, additional knowledge about the current coach of the Baltimore Ravens can be used to answer the question. {answer = None}"""),
        dict(role='user', content="""Q: Rift Valley Province is located in a nation that uses which form of currency?\nKnowledge Triplets: Rift Valley Province, location.administrative_division.country, Kenya\nRift Valley Province, location.location.geolocation, UnName_Entity\nRift Valley Province, location.mailing_address.state_province_region, UnName_Entity\nKenya, location.country.currency_used, Kenyan shilling\nA: """),
        dict(role='assistant', content="""{sufficient = True}. Based on the given knowledge triplets, Rift Valley Province is located in Kenya, which uses the Kenyan shilling as its currency. Therefore, the answer to the question is {answer = "Kenyan shilling"}."""),
        dict(role='user', content=f"""Q: {question}\nKnowledge Triplets: {flatten_chains}\nA: """)
    ]
    return prompts


def answer_prompt(question: str, triple_chains: List[List[List[str]]]):
    """
    :param question: QA question
    :type question: str
    :param triple_chains: shape = (beam, chain, triple, element)
    :type triple_chains: list
    """
    flatten_chains = '\n'.join(['\n'.join([', '.join([e for e in triple]) for triple in triples]) for chain in triple_chains for triples in chain])
    prompts = [
        dict(role='system', content="""You are an AI assistant that helps people find information and answer questions using available knowledge triplets."""),
        dict(role='user', content="""Given a question and the associated retrieved knowledge triplets (entity, relation, entity), you are asked to answer the question with these triplets and your knowledge. Please output the answer like {answer = <your answer>} if you know the answer. Otherwise please output {answer = None}.\nQ: Find the person who said "Taste cannot be controlled by law", what did this person die from?\nKnowledge Triplets: Taste cannot be controlled by law., media_common.quotation.author, Thomas Jefferson\nA: """),
        dict(role='assistant', content="""Based on the given knowledge triplets, it's not sufficient to answer the entire question. The triplets only provide information about the person who said "Taste cannot be controlled by law," which is Thomas Jefferson. To answer the second part of the question, it's necessary to have additional knowledge about where Thomas Jefferson's dead. {answer = None}"""),
        dict(role='user', content="""Q: The artist nominated for The Long Winter lived where?\nKnowledge Triplets: The Long Winter, book.written_work.author, Laura Ingalls Wilder\nLaura Ingalls Wilder, people.person.places_lived, Unknown-Entity\nUnknown-Entity, people.place_lived.location, De Smet\nA: """),
        dict(role='assistant', content="""Based on the given knowledge triplets, the author of The Long Winter, Laura Ingalls Wilder, lived in De Smet. Therefore, the answer to the question is {answer = "De Smet"}."""),
        dict(role='user', content="""Q: Who is the coach of the team owned by Steve Bisciotti?\nKnowledge Triplets: Steve Bisciotti, sports.professional_sports_team.owner_s, Baltimore Ravens\nSteve Bisciotti, sports.sports_team_owner.teams_owned, Baltimore Ravens\nSteve Bisciotti, organization.organization_founder.organizations_founded, Allegis Group\nA: """),
        dict(role='assistant', content="""Based on the given knowledge triplets, the coach of the team owned by Steve Bisciotti is not explicitly mentioned. However, it can be inferred that the team owned by Steve Bisciotti is the Baltimore Ravens, a professional sports team. Therefore, additional knowledge about the current coach of the Baltimore Ravens can be used to answer the question. {answer = None}"""),
        dict(role='user', content="""Q: Rift Valley Province is located in a nation that uses which form of currency?\nKnowledge Triplets: Rift Valley Province, location.administrative_division.country, Kenya\nRift Valley Province, location.location.geolocation, UnName_Entity\nRift Valley Province, location.mailing_address.state_province_region, UnName_Entity\nKenya, location.country.currency_used, Kenyan shilling\nA: """),
        dict(role='assistant', content="""Based on the given knowledge triplets, Rift Valley Province is located in Kenya, which uses the Kenyan shilling as its currency. Therefore, the answer to the question is {answer = "Kenyan shilling"}."""),
        dict(role='user', content=f"""Q: {question}\nKnowledge Triplets: {flatten_chains}\nA: """)
    ]
    return prompts


def entity_recognition_prompt(question: str, width: int):
    prompts = [
        dict(role='system', content="You are an AI assistant that helps people find useful information in questions."),
        dict(role='user', content=f"Given a question, you are asked to retrieve the named entities in the question. Please output the most probable entities like {{entities: [<entity 1>, <entity 2>, ...]}} with no more than {width} entities.\nQ: Find the person who said 'Taste cannot be controlled by law'."),
        dict(role='assistant', content="{entities: ['Taste cannot be controlled by law']}"),
        dict(role='user', content="Q: Which of Anne Hathaway's parents was in the movie 'The Princess Diaries 2: Royal Engagement'?"),
        dict(role='assistant', content="{entities: ['Anne Hathaway', 'The Princess Diaries 2: Royal Engagement']}"),
        dict(role='user', content=f"Q: {question}"),
    ]
    return prompts