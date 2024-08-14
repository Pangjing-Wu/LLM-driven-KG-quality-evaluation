import abc
import re
from typing import Dict, List, Optional

from rdflib import Graph
from SPARQLWrapper import SPARQLWrapper, JSON


class BaseSPARQL(abc.ABC):
    
    def __init__(self) -> None:
        raise NotImplementedError

    def get_all_head_entity(self) -> str:
        raise NotImplementedError
    
    def get_all_tail_entity(self) -> str:
        raise NotImplementedError
    
    def get_all_relations(self) -> str:
        raise NotImplementedError
    
    def get_head_entity(self, relation: str, tail_entity: str) -> str:
        raise NotImplementedError

    def get_tail_entity(self, head_entity: str, relation: str) -> str:
        raise NotImplementedError

    def get_relation_by_head(self, head_entity: str) -> str:
        raise NotImplementedError

    def get_relation_by_tail(self, tail_entity: str) -> str:
        raise NotImplementedError


class SPARQLService(BaseSPARQL):
    
    def __init__(
        self, 
        service_path: str = "http://localhost:8890/sparql", 
        domain: Optional[str] = None
        ) -> None:
        self.sparql = SPARQLWrapper(service_path)
        self.domain = domain
        
    def __call__(self, query: str) -> List[Dict[str, str]]:
        return self.__query(query)
    
    def get_all_head_entity(self) -> List[str]:
        query = 'SELECT ?h\nWHERE {\n?h ?r ?t  .\n}'
        entities = self.__query(query)
        return [entity['h']['value'] for entity in entities]
    
    def get_all_tail_entity(self) -> List[str]:
        query = 'SELECT ?t\nWHERE {\n?h ?r ?t  .\n}'
        entities = self.__query(query)
        return [entity['t']['value'] for entity in entities]
    
    def get_all_relations(self) -> str:
        query = 'SELECT ?r\nWHERE {\n?h ?r ?t  .\n}'
        entities = self.__query(query)
        return [entity['r']['value'] for entity in entities]
    
    def get_head_entity(self, relation: str, tail_entity: str) -> List[str]:
        tail_entity = re.sub(self.domain, '', tail_entity)
        relation    = re.sub(self.domain, '', relation)
        query = f'PREFIX ns: <{self.domain}>\n' if self.domain else ''
        query += 'SELECT ?h\nWHERE {\n?h ns:%s ns:%s  .\n}' % (relation, tail_entity)
        entities = self.__query(query)
        return [entity['h']['value'] for entity in entities]
    
    def get_tail_entity(self, head_entity: str, relation: str) -> List[str]:
        head_entity = re.sub(self.domain, '', head_entity)
        relation    = re.sub(self.domain, '', relation)
        query = f'PREFIX ns: <{self.domain}>\n' if self.domain else ''
        query += 'SELECT ?t\nWHERE {\nns:%s ns:%s ?t .\n}' % (head_entity, relation)
        entities = self.__query(query)
        return [entity['t']['value'] for entity in entities]
    
    def get_relation_by_head(self, head_entity: str) -> List[str]:
        head_entity = re.sub(self.domain, '', head_entity)
        query = f'PREFIX ns: <{self.domain}>\n' if self.domain else ''
        query += 'SELECT ?r\nWHERE {\nns:%s ?r ?t .\n}' % head_entity
        relations = self.__query(query)
        return [relation['r']['value'] for relation in relations]
    
    def get_relation_by_tail(self, tail_entity: str) -> List[str]:
        tail_entity = re.sub(self.domain, '', tail_entity)
        query = f'PREFIX ns: <{self.domain}>\n' if self.domain else ''
        query += 'SELECT ?r\nWHERE {\n?h ?r ns:%s .\n}' % tail_entity
        relations = self.__query(query)
        return [relation['r']['value'] for relation in relations]

    def __query(self, query: str):
        self.sparql.setQuery(query)
        self.sparql.setReturnFormat(JSON)
        return self.sparql.query().convert()["results"]["bindings"]
    
    
class RDFLibSPARQL(BaseSPARQL):
    
    def __init__(
        self,
        path: str,
        format: str = 'nt',
        domain: Optional[str] = None
        ) -> None:
        self.kg = Graph().parse(path, format=format)
        self.domain = domain
        
    def __call__(self, query: str) -> List[Dict[str, str]]:
        return self.kg.query(query)
    
    def get_all_head_entity(self) -> List[str]:
        query = 'SELECT ?h\nWHERE {\n?h ?r ?t  .\n}'
        return [str(row.h) for row in self.kg.query(query)]
    
    def get_all_tail_entity(self) -> List[str]:
        query = 'SELECT ?t\nWHERE {\n?h ?r ?t  .\n}'
        return [str(row.t) for row in self.kg.query(query)]
    
    def get_all_relations(self) -> str:
        query = 'SELECT ?r\nWHERE {\n?h ?r ?t  .\n}'
        return [str(row.r) for row in self.kg.query(query)]
    
    def get_head_entity(self, relation: str, tail_entity: str) -> List[str]:
        # tail_entity = re.sub(self.domain, '', tail_entity)
        # relation    = re.sub(self.domain, '', relation)
        try:
            query = f'PREFIX ns: <{self.domain}>\n' if self.domain else ''
            query += 'SELECT ?h\nWHERE {\n?h ns:%s ns:%s .\n}' % (relation, tail_entity)
            return [str(row.h) for row in self.kg.query(query)]
        except Exception as e:
            with open('./debug.log', 'w') as f:
                f.write(query)
            raise e
    
    def get_tail_entity(self, head_entity: str, relation: str) -> List[str]:
        # tail_entity = re.sub(self.domain, '', tail_entity)
        # relation    = re.sub(self.domain, '', relation)
        try:
            query = f'PREFIX ns: <{self.domain}>\n' if self.domain else ''
            query += 'SELECT ?t\nWHERE {\nns:%s ns:%s ?t .\n}' % (head_entity, relation)
            return [str(row.t) for row in self.kg.query(query)]
        except Exception as e:
            with open('./debug.log', 'w') as f:
                f.write(query)
            raise e
        
    def get_relation_by_head(self, head_entity: str) -> List[str]:
        try:
            query = f'PREFIX ns: <{self.domain}>\n' if self.domain else ''
            query += 'SELECT ?r\nWHERE {\nns:%s ?r ?t .\n}' % head_entity
            return [str(row.r) for row in self.kg.query(query)]
        except Exception as e:
            with open('./debug.log', 'w') as f:
                f.write(query)
            raise e
    
    def get_relation_by_tail(self, tail_entity: str) -> List[str]:
        try:
            query = f'PREFIX ns: <{self.domain}>\n' if self.domain else ''
            query += 'SELECT ?r\nWHERE {\n?h ?r ns:%s .\n}' % tail_entity
            return [str(row.r) for row in self.kg.query(query)]
        except Exception as e:
            with open('./debug.log', 'w') as f:
                f.write(query)
            raise e