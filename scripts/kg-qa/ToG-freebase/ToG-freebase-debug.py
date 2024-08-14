import argparse
import json
import os
import random
import re

import requests
import transformers
import torch
from SPARQLWrapper import SPARQLWrapper, JSON
from tqdm import tqdm
from transformers import AutoTokenizer, BitsAndBytesConfig

SPARQLPATH = "http://localhost:8890/sparql"


random.seed(0)
torch.manual_seed(0)


global llm_chat
device = 'cuda:1'

class LLAMA(object):
    
    def __init__(self, model: str, quantization=False) -> None:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model_kwargs = dict(torch_dtype=torch.bfloat16)
        if quantization:
            model_kwargs.update(dict(quantization_config=bnb_config))
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=f'meta-llama/{model}',
            model_kwargs=model_kwargs,
            device_map=device
        )
    
    def __call__(self, prompt, temperature, max_tokens):
        terminators = [
            self.pipeline.tokenizer.eos_token_id,
            # pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>") # uncomment if you chat with LLAMA-3.
        ]
        
        temperature = temperature if temperature else None
        do_sample   = True if temperature else False
        top_p       = 0.95 if temperature else None
        top_k       = 50 if temperature else None
        
        output = self.pipeline(
            prompt,
            max_new_tokens=max_tokens,
            eos_token_id=terminators,
            pad_token_id=self.pipeline.tokenizer.eos_token_id,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            return_full_text=False)
        return output[0]["generated_text"]
    

class GLM4(LLAMA):
    
    def __init__(self, model: str, quantization=False) -> None:
        model_kwargs = dict(torch_dtype=torch.bfloat16)
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=f'THUDM/{model}',
            model_kwargs=model_kwargs,
            device_map=device,
            trust_remote_code=True
        )
        setattr(self.pipeline, 'tokenizer', AutoTokenizer.from_pretrained(f'THUDM/{model}', trust_remote_code=True))


class AzureChatGPT(object):
    base = 'https://comp.azure-api.net/azure'
    
    def __init__(self, model: str) -> None:
        self.model = model
        self.key   = os.environ['AZURE_GPT_KEY']

    def __call__(self, prompt, temperature, max_tokens):
        body     = dict(messages=[dict(content=prompt, role='user')], temperature=temperature, max_tokens=max_tokens)
        response = requests.post(f'{self.base}/openai/deployments/{self.model}/chat/completions', json=body, headers={'api-key': self.key})
        outputs  = response.json()
        try:
            return outputs["choices"][0]["message"]["content"]
        except KeyError:
            raise RuntimeError(f'capture unexpected response while invoking the API, please check the response below:\n{outputs}')


def execurte_sparql(sparql_txt):
    sparql = SPARQLWrapper(SPARQLPATH)
    sparql.setQuery(sparql_txt)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    return results["results"]["bindings"]

def replace_relation_prefix(relations):
    return [relation['relation']['value'].replace("http://rdf.freebase.com/ns/","") for relation in relations]

def abandon_rels(relation):
    if relation == "type.object.type" or relation == "type.object.name" or relation.startswith("common.") or relation.startswith("freebase.") or "sameAs" in relation:
        return True

sparql_head_relations = """\nPREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT ?relation\nWHERE {\n  ns:%s ?relation ?x .\n}"""
sparql_tail_relations = """\nPREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT ?relation\nWHERE {\n  ?x ?relation ns:%s .\n}"""

def construct_relation_prune_prompt(question, entity_name, total_relations, args):
    extract_relation_prompt = """Please retrieve %s relations (separated by semicolon) that contribute to the question and rate their contribution on a scale from 0 to 1 (the sum of the scores of %s relations is 1).
        Q: Name the president of the country whose main spoken language was Brahui in 1980?
        Topic Entity: Brahui Language
        Relations: language.human_language.main_country; language.human_language.language_family; language.human_language.iso_639_3_code; base.rosetta.languoid.parent; language.human_language.writing_system; base.rosetta.languoid.languoid_class; language.human_language.countries_spoken_in; kg.object_profile.prominent_type; base.rosetta.languoid.document; base.ontologies.ontology_instance.equivalent_instances; base.rosetta.languoid.local_name; language.human_language.region
        A: 1. {language.human_language.main_country (Score: 0.4))}: This relation is highly relevant as it directly relates to the country whose president is being asked for, and the main country where Brahui language is spoken in 1980.
        2. {language.human_language.countries_spoken_in (Score: 0.3)}: This relation is also relevant as it provides information on the countries where Brahui language is spoken, which could help narrow down the search for the president.
        3. {base.rosetta.languoid.parent (Score: 0.2)}: This relation is less relevant but still provides some context on the language family to which Brahui belongs, which could be useful in understanding the linguistic and cultural background of the country in question.

        Q: """
    return extract_relation_prompt % (args.width, args.width) + question + '\nTopic Entity: ' + entity_name + '\nRelations: '+ '; '.join(total_relations) + "\nA: "

def clean_relations(string, entity_id, head_relations):
    pattern = r"{\s*(?P<relation>[^()]+)\s+\(Score:\s+(?P<score>[0-9.]+)\)}"
    relations=[]
    for match in re.finditer(pattern, string):
        relation = match.group("relation").strip()
        relation = re.sub(r'[{},?!#@]', '', relation)
        if ';' in relation:
            continue
        score = match.group("score")
        if not relation or not score:
            return False, "output uncompleted.."
        try:
            score = float(score)
        except ValueError:
            return False, "Invalid score"
        if relation in head_relations:
            relations.append({"entity": entity_id, "relation": relation, "score": score, "head": True})
        else:
            relations.append({"entity": entity_id, "relation": relation, "score": score, "head": False})
    if not relations:
        return False, "No relations found"
    return True, relations

def relation_search_prune(entity_id, entity_name, pre_relations, pre_head, question, args):
    sparql_relations_extract_head = sparql_head_relations % (entity_id)
    head_relations = execurte_sparql(sparql_relations_extract_head)
    head_relations = replace_relation_prefix(head_relations)
    
    sparql_relations_extract_tail= sparql_tail_relations % (entity_id)
    tail_relations = execurte_sparql(sparql_relations_extract_tail)
    tail_relations = replace_relation_prefix(tail_relations)
    
    if args.remove_unnecessary_rel:
        head_relations = [relation for relation in head_relations if not abandon_rels(relation)]
        tail_relations = [relation for relation in tail_relations if not abandon_rels(relation)]
    

    if len(pre_relations) != 0 and pre_head != -1:
        tail_relations = [rel for rel in tail_relations if not pre_head or rel not in pre_relations]
        head_relations = [rel for rel in head_relations if pre_head or rel not in pre_relations]

    head_relations = list(set(head_relations))
    tail_relations = list(set(tail_relations))
    total_relations = head_relations+tail_relations
    total_relations = random.sample(total_relations, args.max_relation_size) if len(total_relations) > args.max_relation_size else total_relations
    total_relations.sort()  # make sure the order in prompt is always equal
    if args.prune_tools == "llm":
        prompt = construct_relation_prune_prompt(question, entity_name, total_relations, args)
        result = llm_chat(prompt, args.temperature_exploration, args.max_length)
        flag, retrieve_relations_with_scores = clean_relations(result, entity_id, head_relations)
        # NOTE: I observed sometime the LLM will keeping generate a hallucinative question with several relations and score the relations. 
        # These relations are noise and sometimes may cause exception.
        # It can be avoid by take the slice on the retrieve_relations.
        # NOTE: I have further observed LLM can still fabricate relations and rank it as top.
        if flag and any(record['relation'] not in total_relations for record in retrieve_relations_with_scores):
            raise ValueError('observed fabricated relations.')

    if flag:
        return retrieve_relations_with_scores
    else:
        return [] # format error or too small max_length
    
    
# def execurte_sparql(sparql_txt):
#     pass

def replace_entities_prefix(entities):
    return [entity['tailEntity']['value'].replace("http://rdf.freebase.com/ns/","") for entity in entities]

sparql_tail_entities_extract = """PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT ?tailEntity\nWHERE {\nns:%s ns:%s ?tailEntity .\n}""" 
sparql_head_entities_extract = """PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT ?tailEntity\nWHERE {\n?tailEntity ns:%s ns:%s  .\n}"""

def entity_search(entity, relation, head=True):
    if head:
        tail_entities_extract = sparql_tail_entities_extract% (entity, relation)
        entities = execurte_sparql(tail_entities_extract)
    else:
        head_entities_extract = sparql_head_entities_extract% (relation, entity)
        entities = execurte_sparql(head_entities_extract)

    entity_ids = replace_entities_prefix(entities)
    new_entity = [entity for entity in entity_ids if entity.startswith("m.")]
    return new_entity

sparql_id = """PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT DISTINCT ?tailEntity\nWHERE {\n  {\n    ?entity ns:type.object.name ?tailEntity .\n    FILTER(?entity = ns:%s)\n  }\n  UNION\n  {\n    ?entity <http://www.w3.org/2002/07/owl#sameAs> ?tailEntity .\n    FILTER(?entity = ns:%s)\n  }\n}"""

def id2entity_name_or_type(entity_id):
    entity_id = sparql_id % (entity_id, entity_id)
    sparql = SPARQLWrapper(SPARQLPATH)
    sparql.setQuery(entity_id)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    if len(results["results"]["bindings"])==0:
        return "UnName_Entity"
    else:
        return results["results"]["bindings"][0]['tailEntity']['value']
    
def all_unknown_entity(entity_candidates):
    return all(candidate == "UnName_Entity" for candidate in entity_candidates)

def del_unknown_entity(entity_candidates):
    if len(entity_candidates)==1 and entity_candidates[0]=="UnName_Entity":
        return entity_candidates
    entity_candidates = [candidate for candidate in entity_candidates if candidate != "UnName_Entity"]
    return entity_candidates

def clean_scores(string, entity_candidates):
    scores = re.findall(r'\d+\.\d+', string)
    try:
        scores = [float(number) for number in scores][:len(entity_candidates)]
    except IndexError:
        print("All entities are created equal.")
        return [1/len(entity_candidates)] * len(entity_candidates)
    else:
        return scores

score_entity_candidates_prompt = """Please score the entities' contribution to the question on a scale from 0 to 1 (the sum of the scores of all entities is 1).
Q: The movie featured Miley Cyrus and was produced by Tobin Armbrust?
Relation: film.producer.film
Entites: The Resident; So Undercover; Let Me In; Begin Again; The Quiet Ones; A Walk Among the Tombstones
Score: 0.0, 1.0, 0.0, 0.0, 0.0, 0.0
The movie that matches the given criteria is "So Undercover" with Miley Cyrus and produced by Tobin Armbrust. Therefore, the score for "So Undercover" would be 1, and the scores for all other entities would be 0.

Q: {}
Relation: {}
Entites: """

def construct_entity_score_prompt(question, relation, entity_candidates):
    return score_entity_candidates_prompt.format(question, relation) + "; ".join(entity_candidates) + '\nScore: '

def entity_score(question, entity_candidates_id, score, relation, args):
    entity_candidates = [id2entity_name_or_type(entity_id) for entity_id in entity_candidates_id]
    if all_unknown_entity(entity_candidates):
        return [1/len(entity_candidates) * score] * len(entity_candidates), entity_candidates, entity_candidates_id
    entity_candidates = del_unknown_entity(entity_candidates)
    if len(entity_candidates) == 1:
        return [score], entity_candidates, entity_candidates_id
    if len(entity_candidates) == 0:
        return [0.0], entity_candidates, entity_candidates_id
    
    # make sure the id and entity are in the same order
    zipped_lists = sorted(zip(entity_candidates, entity_candidates_id))
    entity_candidates, entity_candidates_id = zip(*zipped_lists)
    entity_candidates = list(entity_candidates)
    entity_candidates_id = list(entity_candidates_id)
    if args.prune_tools == "llm":
        prompt = construct_entity_score_prompt(question, relation, entity_candidates)

        result = llm_chat(prompt, args.temperature_exploration, args.max_length)
        return [float(x) * score for x in clean_scores(result, entity_candidates)], entity_candidates, entity_candidates_id


def update_history(entity_candidates, entity, scores, entity_candidates_id, total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head):
    if len(entity_candidates) == 0:
        entity_candidates.append("[FINISH]")
        entity_candidates_id = ["[FINISH_ID]"]
    candidates_relation = [entity['relation']] * len(entity_candidates)
    topic_entities = [entity['entity']] * len(entity_candidates)
    head_num = [entity['head']] * len(entity_candidates)
    total_candidates.extend(entity_candidates)
    total_scores.extend(scores)
    total_relations.extend(candidates_relation)
    total_entities_id.extend(entity_candidates_id)
    total_topic_entities.extend(topic_entities)
    total_head.extend(head_num)
    return total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head


def entity_prune(total_entities_id, total_relations, total_candidates, total_topic_entities, total_head, total_scores, args):
    zipped = list(zip(total_entities_id, total_relations, total_candidates, total_topic_entities, total_head, total_scores))
    sorted_zipped = sorted(zipped, key=lambda x: x[5], reverse=True)
    sorted_entities_id, sorted_relations, sorted_candidates, sorted_topic_entities, sorted_head, sorted_scores = [x[0] for x in sorted_zipped], [x[1] for x in sorted_zipped], [x[2] for x in sorted_zipped], [x[3] for x in sorted_zipped], [x[4] for x in sorted_zipped], [x[5] for x in sorted_zipped]

    entities_id, relations, candidates, topics, heads, scores = sorted_entities_id[:args.width], sorted_relations[:args.width], sorted_candidates[:args.width], sorted_topic_entities[:args.width], sorted_head[:args.width], sorted_scores[:args.width]
    merged_list = list(zip(entities_id, relations, candidates, topics, heads, scores))
    filtered_list = [(id, rel, ent, top, hea, score) for id, rel, ent, top, hea, score in merged_list if score != 0]
    if len(filtered_list) ==0:
        return False, [], [], [], []
    entities_id, relations, candidates, tops, heads, scores = map(list, zip(*filtered_list))

    tops = [id2entity_name_or_type(entity_id) for entity_id in tops]
    cluster_chain_of_entities = [[(tops[i], relations[i], candidates[i]) for i in range(len(candidates))]]
    return True, cluster_chain_of_entities, entities_id, relations, heads


def extract_answer(text):
    start_index = text.find("{")
    end_index = text.find("}")
    if start_index != -1 and end_index != -1:
        return text[start_index+1:end_index].strip()
    else:
        return ""

def if_true(prompt):
    if prompt.lower().strip().replace(" ","")=="yes":
        return True
    return False

prompt_evaluate="""Given a question and the associated retrieved knowledge graph triplets (entity, relation, entity), you are asked to answer whether it's sufficient for you to answer the question with these triplets and your knowledge (Yes or No).
Q: Find the person who said \"Taste cannot be controlled by law\", what did this person die from?
Knowledge Triplets: Taste cannot be controlled by law., media_common.quotation.author, Thomas Jefferson
A: {No}. Based on the given knowledge triplets, it's not sufficient to answer the entire question. The triplets only provide information about the person who said "Taste cannot be controlled by law," which is Thomas Jefferson. To answer the second part of the question, it's necessary to have additional knowledge about where Thomas Jefferson's dead.

Q: The artist nominated for The Long Winter lived where?
Knowledge Triplets: The Long Winter, book.written_work.author, Laura Ingalls Wilder
Laura Ingalls Wilder, people.person.places_lived, Unknown-Entity
Unknown-Entity, people.place_lived.location, De Smet
A: {Yes}. Based on the given knowledge triplets, the author of The Long Winter, Laura Ingalls Wilder, lived in De Smet. Therefore, the answer to the question is {De Smet}.

Q: Who is the coach of the team owned by Steve Bisciotti?
Knowledge Triplets: Steve Bisciotti, sports.professional_sports_team.owner_s, Baltimore Ravens
Steve Bisciotti, sports.sports_team_owner.teams_owned, Baltimore Ravens
Steve Bisciotti, organization.organization_founder.organizations_founded, Allegis Group
A: {No}. Based on the given knowledge triplets, the coach of the team owned by Steve Bisciotti is not explicitly mentioned. However, it can be inferred that the team owned by Steve Bisciotti is the Baltimore Ravens, a professional sports team. Therefore, additional knowledge about the current coach of the Baltimore Ravens can be used to answer the question.

Q: Rift Valley Province is located in a nation that uses which form of currency?
Knowledge Triplets: Rift Valley Province, location.administrative_division.country, Kenya
Rift Valley Province, location.location.geolocation, UnName_Entity
Rift Valley Province, location.mailing_address.state_province_region, UnName_Entity
Kenya, location.country.currency_used, Kenyan shilling
A: {Yes}. Based on the given knowledge triplets, Rift Valley Province is located in Kenya, which uses the Kenyan shilling as its currency. Therefore, the answer to the question is {Kenyan shilling}.

Q: The country with the National Anthem of Bolivia borders which nations?
Knowledge Triplets: National Anthem of Bolivia, government.national_anthem_of_a_country.anthem, UnName_Entity
National Anthem of Bolivia, music.composition.composer, Leopoldo Benedetto Vincenti
National Anthem of Bolivia, music.composition.lyricist, José Ignacio de Sanjinés
UnName_Entity, government.national_anthem_of_a_country.country, Bolivia
Bolivia, location.country.national_anthem, UnName_Entity
A: {No}. Based on the given knowledge triplets, we can infer that the National Anthem of Bolivia is the anthem of Bolivia. Therefore, the country with the National Anthem of Bolivia is Bolivia itself. However, the given knowledge triplets do not provide information about which nations border Bolivia. To answer this question, we need additional knowledge about the geography of Bolivia and its neighboring countries.
"""

def reasoning(question, cluster_chain_of_entities, args):
    prompt = prompt_evaluate + question
    chain_prompt = '\n'.join([', '.join([str(x) for x in chain]) for sublist in cluster_chain_of_entities for chain in sublist])
    prompt += "\nKnowledge Triplets: " + chain_prompt + 'A: '

    response = llm_chat(prompt, args.temperature_reasoning, args.max_length)
    
    result = extract_answer(response)
    if if_true(result):
        return True, response
    else:
        return False, response
    
    
answer_prompt = """Given a question and the associated retrieved knowledge graph triplets (entity, relation, entity), you are asked to answer the question with these triplets and your knowledge.
Q: Find the person who said \"Taste cannot be controlled by law\", what did this person die from?
Knowledge Triplets: Taste cannot be controlled by law., media_common.quotation.author, Thomas Jefferson
A: Based on the given knowledge triplets, it's not sufficient to answer the entire question. The triplets only provide information about the person who said "Taste cannot be controlled by law," which is Thomas Jefferson. To answer the second part of the question, it's necessary to have additional knowledge about where Thomas Jefferson's dead.

Q: The artist nominated for The Long Winter lived where?
Knowledge Triplets: The Long Winter, book.written_work.author, Laura Ingalls Wilder
Laura Ingalls Wilder, people.person.places_lived, Unknown-Entity
Unknown-Entity, people.place_lived.location, De Smet
A: Based on the given knowledge triplets, the author of The Long Winter, Laura Ingalls Wilder, lived in De Smet. Therefore, the answer to the question is {De Smet}.

Q: Who is the coach of the team owned by Steve Bisciotti?
Knowledge Triplets: Steve Bisciotti, sports.professional_sports_team.owner_s, Baltimore Ravens
Steve Bisciotti, sports.sports_team_owner.teams_owned, Baltimore Ravens
Steve Bisciotti, organization.organization_founder.organizations_founded, Allegis Group
A: Based on the given knowledge triplets, the coach of the team owned by Steve Bisciotti is not explicitly mentioned. However, it can be inferred that the team owned by Steve Bisciotti is the Baltimore Ravens, a professional sports team. Therefore, additional knowledge about the current coach of the Baltimore Ravens can be used to answer the question.

Q: Rift Valley Province is located in a nation that uses which form of currency?
Knowledge Triplets: Rift Valley Province, location.administrative_division.country, Kenya
Rift Valley Province, location.location.geolocation, UnName_Entity
Rift Valley Province, location.mailing_address.state_province_region, UnName_Entity
Kenya, location.country.currency_used, Kenyan shilling
A: Based on the given knowledge triplets, Rift Valley Province is located in Kenya, which uses the Kenyan shilling as its currency. Therefore, the answer to the question is {Kenyan shilling}.

Q: The country with the National Anthem of Bolivia borders which nations?
Knowledge Triplets: National Anthem of Bolivia, government.national_anthem_of_a_country.anthem, UnName_Entity
National Anthem of Bolivia, music.composition.composer, Leopoldo Benedetto Vincenti
National Anthem of Bolivia, music.composition.lyricist, José Ignacio de Sanjinés
UnName_Entity, government.national_anthem_of_a_country.country, Bolivia
Bolivia, location.country.national_anthem, UnName_Entity
A: Based on the given knowledge triplets, we can infer that the National Anthem of Bolivia is the anthem of Bolivia. Therefore, the country with the National Anthem of Bolivia is Bolivia itself. However, the given knowledge triplets do not provide information about which nations border Bolivia. To answer this question, we need additional knowledge about the geography of Bolivia and its neighboring countries.

Q: {}
"""

def generate_answer(question, cluster_chain_of_entities, args): 
    prompt = answer_prompt + question + '\n'
    chain_prompt = '\n'.join([', '.join([str(x) for x in chain]) for sublist in cluster_chain_of_entities for chain in sublist])
    prompt += "\nKnowledge Triplets: " + chain_prompt + 'A: '
    result = llm_chat(prompt, args.temperature_reasoning, args.max_length)
    return result

def save_2_jsonl(question, answer, cluster_chain_of_entities, filepath):
    dict = {"question":question, "results": answer, "reasoning_chains": cluster_chain_of_entities}
    with open(filepath, "a") as outfile:
        json_str = json.dumps(dict)
        outfile.write(json_str + "\n")

def half_stop(question, cluster_chain_of_entities, filepath, args):
    print("No new knowledge added during search depth %d, stop searching." % args.depth)
    answer = generate_answer(question, cluster_chain_of_entities, args)
    save_2_jsonl(question, answer, cluster_chain_of_entities, filepath)
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process the arguments for the dataset and model settings.')
    parser.add_argument('--dataset', type=str, default='cwq', help='Dataset to use')
    parser.add_argument('--llm', default='Llama-2-13b-chat-hf', type=str, help='LLM to use')
    parser.add_argument('--question_string', type=str, default='question', help='key of questions')
    parser.add_argument('--max_length', type=int, default=256, help='Maximum length of the query')
    parser.add_argument('--temperature_exploration', type=float, default=0.4, help='Temperature for exploration')
    parser.add_argument('--temperature_reasoning', type=float, default=0.0, help='Temperature for reasoning')
    parser.add_argument('--width', type=int, default=3, help='Width parameter for the search')
    parser.add_argument('--depth', type=int, default=3, help='Depth of the reasoning')
    parser.add_argument('--remove_unnecessary_rel', action='store_true', help='Flag to remove unnecessary relations')
    parser.add_argument('--num_retain_entity', type=int, default=5, help='Number of entities to retain')
    parser.add_argument('--prune_tools', type=str, default='llm', help='Tool to use for pruning')
    parser.add_argument('--max_relation_size', type=int, default=200, help='Maximum size of relations')
    parser.add_argument('--quantization', '-q', action='store_true', help='use 4-bit quantization')

    args = parser.parse_args()
    print('\n'.join([f'{k} = {str(v)}' for k, v in sorted(dict(vars(args)).items())]))
    
    filepath = f'./results/ToG-freebase/ToG-debug-{args.dataset}.jsonl'
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(f'./data/{args.dataset}.json',encoding='utf-8') as f:
        datas = json.load(f)

    if 'llama' in args.llm.lower():
        llm_chat = LLAMA(args.llm, args.quantization)
    elif 'gpt' in args.llm.lower():
        llm_chat = AzureChatGPT(args.llm)
    
    for data in tqdm(datas):
        try:
            question = data[args.question_string]
            print(f'\n{question = }')
            topic_entity = data['topic_entity']
            cluster_chain_of_entities = []
            pre_relations = [], 
            pre_heads= [-1] * len(topic_entity)
            flag_printed = False
            make_answer = False
            fabricate = False
            for depth in range(1, args.depth+1):
                if make_answer:
                    break
                current_entity_relations_list = []
                i=0
                for entity in topic_entity:
                    if entity!="[FINISH_ID]":
                        try:
                            retrieve_relations_with_scores = relation_search_prune(entity, topic_entity[entity], pre_relations, pre_heads[i], question, args)  # best entity triplet, entitiy_id
                        except ValueError as e:
                            print('observed fabricate relation, stop answering this question')
                            make_answer = half_stop(question, cluster_chain_of_entities, filepath, args)
                            fabricate = True
                            break
                        else:
                            current_entity_relations_list.extend(retrieve_relations_with_scores)
                    i+=1
                if fabricate:
                    break
                
                total_candidates = []
                total_scores = []
                total_relations = []
                total_entities_id = []
                total_topic_entities = []
                total_head = []

                for entity in current_entity_relations_list:
                    if entity['head']:
                        entity_candidates_id = entity_search(entity['entity'], entity['relation'], True)
                    else:
                        entity_candidates_id = entity_search(entity['entity'], entity['relation'], False)
                    
                    if len(entity_candidates_id) >=20:
                        entity_candidates_id = random.sample(entity_candidates_id, args.num_retain_entity)

                    if len(entity_candidates_id) ==0:
                        continue

                    scores, entity_candidates, entity_candidates_id = entity_score(question, entity_candidates_id, entity['score'], entity['relation'], args)
                    
                    total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head = update_history(entity_candidates, entity, scores, entity_candidates_id, total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head)
                
                if len(total_candidates) ==0:
                    make_answer = half_stop(question, cluster_chain_of_entities, filepath, args)
                    break
                    
                flag, chain_of_entities, entities_id, pre_relations, pre_heads = entity_prune(total_entities_id, total_relations, total_candidates, total_topic_entities, total_head, total_scores, args)
                cluster_chain_of_entities.append(chain_of_entities)
                if flag:
                    stop, results = reasoning(question, cluster_chain_of_entities, args)
                    if stop:
                        print("ToG stoped at depth %d." % depth)
                        save_2_jsonl(question, results, cluster_chain_of_entities, filepath=filepath)
                        break
                    else:
                        print("depth %d still not find the answer." % depth)
                        topic_entity = {entity: id2entity_name_or_type(entity) for entity in entities_id}
                        if depth >= args.depth:
                            make_answer = half_stop(question, cluster_chain_of_entities, filepath, args)
                else:
                    make_answer = half_stop(question, cluster_chain_of_entities, filepath, args)
                    break
        except Exception as e:
            print(f'catch exception {e}, skip the question {question}.')
            save_2_jsonl(question, '', [], filepath=filepath)