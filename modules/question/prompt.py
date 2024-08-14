from typing import Dict, List, Union


def start_entity_prune_prompt(history: List[Dict[str, Union[str,bool]]], candidate_name: List[str], k: int):
    history = '\n'.join(map(str, history))
    prompts = [
        dict(role='system', content=f"You are an AI assistant that evaluates respondents' answers to identify gaps in their knowledge. Select and list the top {k} knowledge points from the candidate list that the respondent may not have mastered."),
        dict(role='user', content=f"Based on the responses provided (where 'question' corresponds to the question and 'correct' indicates whether the respondent answered correctly or not), select the top {k} knowledge points from the candidate list that the respondent may not have mastered. Output the result in list format.\n\nResponse: {history}\n\nCandidate List: {candidate_name}")
    ]
    return prompts


def question_for_tail_prompt(reasoning_path: List[str]):
    ref = reasoning_path[0]
    ans = reasoning_path[-1]
    prompts = [
        dict(role='system', content="You are an AI assistant tasked with generating a question based on a given knowledge chain."),
        dict(role='user', content=f"""Please generate a question about the entity '{ans}' based on the following knowledge chain: {reasoning_path}. Ensure the question includes the concept entity '{ref}' and summarizes the relationship between '{ref}' and '{ans}' to provide sufficient context for answering. Do not reveal '{ans}' in the question. Format the result as: {{question: <question string>}}."""),
    ]
    return prompts


def question_for_relation_prompt(reasoning_path: List[str]):
    ref = reasoning_path[0]
    ans = reasoning_path[-2]
    last_entity = reasoning_path[-1]
    prompts = [
        dict(role='system', content="You are an AI assistant tasked with generating a question based on a given knowledge chain."),
        dict(role='user', content=f"""Please generate a question that asks about the relationship '{ans}' between '{ref}' and '{last_entity}' based on the following knowledge chain: {reasoning_path}. Ensure the question includes the concept entity '{ref}' and does not reveal '{ans}'. Format the result as: {{question: <question string>}}."""),
    ]
    return prompts


def question_direct_answer_check_prompt(question: str):
    prompts = [
        dict(role='system', content="You are an AI assistant that helps answer users' questions."),
        dict(role='user', content=f"""Please directly answer the question and format your response as {{answer = <your answer>}} if you know the answer. Otherwise, output `None`.\nQuestion: {question}"""),
    ]
    return prompts


def question_answerable_check_prompt(question: str, triple_chains: List[str]):
    flatten_chains = "; ".join(triple_chains)
    prompts = [
        dict(role='system', content="You are an AI assistant that helps people find information and answer questions using available knowledge triplets."),
        dict(role='user', content=f"""Given a question and the relevant knowledge triple chain, please answer the question based on the triple chain and format your response as {{answer: <your answer>}} if you know the answer. Otherwise, output `None`.\nQuestion: {question}\nKnowledge Triple Chain: {flatten_chains}\nThe answer is:"""),
    ]
    return prompts