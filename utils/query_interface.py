# -*- coding: utf-8 -*-
# !/usr/bin/python

"""
# @Time    : 2021/8/31
# @Author  : Yongrui Chen
# @File    : query_interface.py
# @Software: PyCharm
"""

import json
import re
from datetime import datetime
from utils.utils import timeout

from SPARQLWrapper import SPARQLWrapper, JSON

# kb_endpoint = "https://dbpedia.org/sparql"

@timeout(2)
def KB_query_with_timeout(_query, kb_endpoint):
    """
    :param _query: sparql query statement
    :return:
    """
    sparql = SPARQLWrapper(kb_endpoint)
    sparql.setQuery(_query)
    sparql.setReturnFormat(JSON)
    # sparql.setTimeout(5)
    response = sparql.query().convert()
    results = parse_query_results(response)
    return results

def KB_query(_query, kb_endpoint):
    """
    :param _query: sparql query statement
    :return:
    """
    sparql = SPARQLWrapper(kb_endpoint)
    sparql.setQuery(_query)
    sparql.setReturnFormat(JSON)
    # sparql.setTimeout(5)
    response = sparql.query().convert()
    results = parse_query_results(response)
    return results

def query_ent_name(x, kb_endpoint):
    query = "PREFIX ns: <http://rdf.freebase.com/ns/> SELECT ?name WHERE { " + x + " ns:type.object.name ?name .}"
    results = KB_query(query, kb_endpoint)
    if len(results) == 0:
        query = "PREFIX ns: <http://rdf.freebase.com/ns/> SELECT ?name WHERE { " + x + " ns:common.topic.alias ?name .}"
        results = KB_query(query, kb_endpoint)
        if len(results) == 0:
            print(x, "does not have name !")
            return x
    name = results[0]["name"]
    return name

def parse_query_results(response):

    if "boolean" in response:  # ASK
        results = [response["boolean"]]
    else:
        if len(response["results"]["bindings"]) > 0 and "callret-0" in response["results"]["bindings"][0]: # COUNT
            results = [int(response['results']['bindings'][0]['callret-0']['value'])]
        else:
            results = []
            for res in response['results']['bindings']:
                res = {k: v["value"] for k, v in res.items()}
                results.append(res)
    return results

def formalize(query):
    p_where = re.compile(r'[{](.*?)[}]', re.S)
    select_clause = query[:query.find("{")].strip(" ")
    select_clause = [x.strip(" ") for x in select_clause.split(" ")]
    select_clause = " ".join([x for x in select_clause if x != ""])
    select_clause = select_clause.replace("DISTINCT COUNT(?uri)", "COUNT(?uri)")

    where_clauses = re.findall(p_where, query)[0]
    where_clauses = where_clauses.strip(" ").strip(".").strip(" ")
    triples = [[y.strip(" ") for y in x.strip(" ").split(" ") if y != ""]
               for x in where_clauses.split(". ")]
    triples = [" ".join(["?x" if y[0] == "?" and y[1] == "x" else y for y in x]) for x in triples]
    where_clause = " . ".join(triples)
    query = select_clause + "{ " + where_clause + " }"
    return query

def query_answers(query, kb_endpoint):
    query = formalize(query)
    sparql = SPARQLWrapper(kb_endpoint)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    # sparql.setTimeout(5)
    response = sparql.query().convert()

    if "ASK" in query:
        results = [str(response["boolean"])]
    elif "COUNT" in query:
        tmp = response["results"]["bindings"]
        assert len(tmp) == 1 and ".1" in tmp[0]
        results = [tmp[0][".1"]["value"]]
    else:
        tmp = response["results"]["bindings"]
        results = [x["uri"]["value"] for x in tmp]
    return results


if __name__ == '__main__':

    query_1 = "ASK WHERE { " \
              "<http://dbpedia.org/resource/James_Watt> <http://dbpedia.org/ontology/field> <http://dbpedia.org/resource/Mechanical_engineering> }"
    query_2 = "SELECT DISTINCT COUNT(?uri) WHERE { " \
              "?x <http://dbpedia.org/property/partner> <http://dbpedia.org/resource/Dolores_del_R\u00edo> . " \
              "?uri <http://dbpedia.org/property/director> ?x  . " \
              "?uri <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://dbpedia.org/ontology/Film>}"
    query_3 = "SELECT DISTINCT ?uri WHERE { " \
              "?x <http://dbpedia.org/property/partner> <http://dbpedia.org/resource/Dolores_del_R\u00edo> . " \
              "?uri <http://dbpedia.org/property/director> ?x  . " \
              "?uri <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://dbpedia.org/ontology/Film>}"

    print(KB_query(query_2, "http://10.201.61.163:8890//sparql"))
    exit()

    start_time = datetime.now()

    query = """
    PREFIX ns: <http://rdf.freebase.com/ns/>
    SELECT ?r1 ?r2 ?r3 ?r4 ?r5 ?r6 ?r7
    WHERE {
      ?x ?r1 ns:m.02h76fz . # Military Conflict
      ?x ?r2 ?from .
      ?x ?r3 ?to .
      ?x ?r4 ?y .
      ?y ?r5 ns:m.09c7w0 .  # United States of America
      ?x ?r6 ?c .
      ?c ?r7 ?num .
      FILTER (xsd:integer(?num) > 116708) .
    }
    """
    results = KB_query(query, kb_endpoint="http://10.201.81.42:8890/sparql")
    print(results)

    print('used time ', datetime.now() - start_time)