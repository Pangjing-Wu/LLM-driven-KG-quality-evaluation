# NOTE: please create a `path.py`` file under the same folder and specify your kgs' path in it.
SPARQLPATH = "http://localhost:8890/sparql"


import os
from .path import kg_path
from modules.utils.kg.mapping import freebase_mapping


freebase_configs = dict(
    domain = "http://rdf.freebase.com/ns/",
    abandon_relations = ["^type.object.type$", "^type.object.name$", "^common.", "^freebase.", "sameAs"],
    abandon_entity_names = ["UnName_Entity"],
    entity_regex = "^m.",
    entity_mapping = freebase_mapping
)

fb15k_configs = dict(
    domain = "http://freebase.com/",
    path = kg_path + '/freebase/FB15k/all.nt',
    abandon_relations = ["^type.object.type$", "^type.object.name$", "^common.", "^freebase.", "sameAs"],
    abandon_entity_names = ["UnName_Entity"],
    entity_mapping = lambda x: x.replace('_', ' ')
)

fb15k237_configs = dict(
    domain = "http://freebase.com/",
    path = kg_path + '/freebase/FB15k-237/all.nt',
    abandon_relations = ["^type.object.type$", "^type.object.name$", "^common.", "^freebase.", "sameAs"],
    abandon_entity_names = ["UnName_Entity"],
    entity_mapping = lambda x: x.replace('_', ' ')
)

fb15k237refine_configs = dict(
    domain = "http://freebase.com/",
    path = kg_path + '/freebase/FB15K237-Refined/all.nt',
    abandon_relations = ["^type.object.type$", "^type.object.name$", "^common.", "^freebase.", "sameAs"],
    abandon_entity_names = ["UnName_Entity"],
    entity_mapping = lambda x: x.replace('_', ' ')
)

umls_configs = dict(
    domain = "http://umls.com/",
    path = kg_path + '/umls/all.nt',
    entity_mapping = lambda x: x.replace('_', ' ')
)

usmle_configs = dict(
    domain = "http://usmle.com/",
    path = kg_path + '/USMLE/all.nt',
    entity_mapping = lambda x: x.replace('_', ' ')
)