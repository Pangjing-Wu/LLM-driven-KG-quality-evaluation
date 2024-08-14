import re

from config.kg import SPARQLPATH
from SPARQLWrapper import SPARQLWrapper, JSON


def freebase_mapping(entity_id):
    entity_id = re.sub(f'^http://rdf.freebase.com/ns/', '', entity_id)
    query = """PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT DISTINCT ?t\nWHERE {\n  {\n    ?h ns:type.object.name ?t .\n    FILTER(?h = ns:%s)\n  }\n  UNION\n  {\n    ?h <http://www.w3.org/2002/07/owl#sameAs> ?t .\n    FILTER(?h = ns:%s)\n  }\n}"""
    query = query % (entity_id, entity_id)
    sparql = SPARQLWrapper(SPARQLPATH)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    if len(results["results"]["bindings"]) == 0:
        return "UnName_Entity"
    else:
        return results["results"]["bindings"][0]['t']['value']
    

def fb15k237_mapping(entity_id):
    entity_id = re.sub(f'^http://freebase.com/', '', entity_id)
    query = """PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT DISTINCT ?t\nWHERE {\n  {\n    ?h ns:type.object.name ?t .\n    FILTER(?h = ns:%s)\n  }\n  UNION\n  {\n    ?h <http://www.w3.org/2002/07/owl#sameAs> ?t .\n    FILTER(?h = ns:%s)\n  }\n}"""
    query = query % (entity_id, entity_id)
    sparql = SPARQLWrapper(SPARQLPATH)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    if len(results["results"]["bindings"]) == 0:
        return "UnName_Entity"
    else:
        return results["results"]["bindings"][0]['t']['value']