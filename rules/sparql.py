# -*- coding: utf-8 -*-
# !/usr/bin/python

"""
# @Time    : 2021/8/31
# @Author  : Yongrui Chen
# @File    : sparql.py
# @Software: PyCharm
"""

import re
import copy
import sys
import json
sys.path.append("..")
from utils.utils import get_content_from_outermost_brackets, get_content_behind_key_word, \
    is_variable, is_entity, is_type, is_value, is_relation, \
    rename, rename_inv, remove_type, combine_mapping, get_inv_op, get_inv_edge, get_inv_name, get_origin_name, \
    simplify_variable_in_filter, big_bracket_pattern, date_pattern_1
from rules.grammar import AbstractQueryGraph, V_CLASS_IDS, E_CLASS_IDS, build_instance_pool_with_gold, \
    AGG_NAME, CMP_NAME, ORD_NAME


class SPARQLParserLCQ(object):
    def __init__(self):
        pass

    def parse_conditions(self, clause):
        """
        parse where clause of the sparql.
        """
        clause = clause.strip(" ")
        tmp = [x.strip(".") for x in clause.split(" ") if x not in [".", "", " "]]
        assert len(tmp) % 3 == 0
        t_id = -1
        triples = []
        for i, x in enumerate(tmp):
            if i % 3 == 0:
                t_id += 1
                triples.append([rename(x, depth=0)])
            elif i % 3 == 2:
                triples[t_id].append(rename(x, depth=0))
            else:
                triples[t_id].append(x)
        return triples

    def parse_sparql(self, sparql):

        is_ask = sparql.find("ASK") != -1
        is_count = sparql.find("COUNT") != -1

        where_clause = re.search(big_bracket_pattern, sparql).group(0)
        where_clause = where_clause.strip("{").strip("}")
        triples = self.parse_conditions(where_clause)

        conds = {"notUnion": [], "union": [], "subQueries": []}
        for s, p, o in triples:
            conds["notUnion"].append(["Triple", [s, p, o]])

        if is_ask:
            conds["notUnion"].append(["Aggregation", [rename("?uri", depth=0), "ASK", triples[0][0]]])

        if is_count:
            conds["notUnion"] = [[_type, [x.replace(rename("?uri", depth=0), rename("?z", depth=0)) for x in triple]]
                                 for _type, triple in conds["notUnion"]]
            conds["notUnion"].append(["Aggregation", [rename("?z", depth=0), "COUNT", rename("?uri", depth=0)]])

        query = {"select": [rename("?uri", depth=0)],
                 "where": conds,
                 "orderBy": None}
        return query


class SPARQLParserWSP(object):
    def __init__(self):
        pass

        self.op_mapping = {
            "LessOrEqual": "<=",
            "GreaterOrEqual": ">=",
            "Equal": "="
        }

    def parse_sparql(self, sparql):

        _conds = []

        topic_ent = sparql["TopicEntityMid"]

        if len(sparql["InferentialChain"]) == 2:
            _conds.append(["Triple",
                           ["ns:" + rename(topic_ent, depth=0),
                            "ns:" + sparql["InferentialChain"][0],
                            rename("?y", depth=0)]])
            _conds.append(["Triple",
                           [rename("?y", depth=0),
                            "ns:" + sparql["InferentialChain"][1],
                            rename("?x", depth=0)]])
            vars_in_chain = [rename("?y", depth=0), rename("?x", depth=0)]
        else:
            _conds.append(["Triple",
                           ["ns:" + rename(topic_ent, depth=0),
                            "ns:" + sparql["InferentialChain"][0],
                            rename("?x", depth=0)]])
            vars_in_chain = [rename("?x", depth=0)]

        for i, constraint in enumerate(sparql["Constraints"]):
            if constraint["SourceNodeIndex"] >= len(vars_in_chain):
                constraint["SourceNodeIndex"] -= 1
            var = vars_in_chain[constraint["SourceNodeIndex"]]
            if constraint["ArgumentType"] == "Entity":
                ent = constraint["Argument"]
                _conds.append(["Triple",
                               [var,
                                "ns:" + constraint["NodePredicate"],
                                "ns:" + rename(ent, depth=0)]])
            elif constraint["ArgumentType"] == "Value":
                is_date = re.match(date_pattern_1, constraint["Argument"]) is not None
                if is_date:
                    op = self.op_mapping[constraint["Operator"]]
                    date_var = rename("?sk" + str(i), depth=0)
                    date = "\"" + constraint["Argument"] + "\"^^xsd:dateTime"
                    _conds.append(["Triple",
                                   [var,
                                    "ns:" + constraint["NodePredicate"],
                                    date_var]])
                    _conds.append(["Comparison",
                                   [date_var,
                                    op,
                                    date]])
                else:
                    val = "\"" + constraint["Argument"] + "\""
                    _conds.append(["Triple",
                                   [var,
                                    "ns:" + constraint["NodePredicate"],
                                    val]])

        if sparql["Order"]:
            if sparql["Order"]["SortOrder"] == "Descending":
                order = "DESC"
            else:
                order = "ASC"

            if sparql["Order"]["ValueType"] == "DateTime":
                type = "xsd:datetime"
            else:
                type = "xsd:float"

            if sparql["Order"]["NodePredicate"]:
                var = vars_in_chain[sparql["Order"]["SourceNodeIndex"]]
                _conds.append(["Triple",
                               [var,
                                "ns:" + sparql["Order"]["NodePredicate"],
                                rename("?o", depth=0)]])
                ord_by = [order, type, rename("?o", depth=0), "\"" + str(sparql["Order"]["Count"]) + "\""]
            else:
                ord_by = [order, type, rename("?x", depth=0), "\"" + str(sparql["Order"]["Count"]) + "\""]
        else:
            ord_by = None


        conds = {"notUnion": [], "union": [], "subQueries": []}
        conds["notUnion"] = [x for x in _conds]

        query = {"select": [rename("?x", depth=0)],
                 "where": conds,
                 "orderBy": ord_by}

        self.simplify_period(query, 0)
        return query

    def simplify_period(self, query, depth):
        """
        In the time constraint, combine "from" relation AND "to" relation into a new "period" relation.
        Avoid there is circle in the query graph

        ?y ns:government.government_position_held.from ?pfrom ;
        ?y ns:government.government_position_held.to ?pto .
        ?x ns:time.event.start_date ?from .
        ?x ns:time.event.end_date ?to .
        FILTER (?to <= ?pto && ?from >= ?pfrom)
        =============================================>
        ?y ns:government.government_position_held.from###ns:government.government_position_held.to ?sub_period .
        ?x ns:time.event.start_date###ns:time.event.end_date ?period .
        FILTER (?period during ?sub_period)

        @param query: parsed query
        @param depth: main query(0), sub query (1) ...
        """
        var_from, var_to = None, None
        index_from, index_to = -1, -1
        # Time constraint must not be union
        # Find the first ?from and ?to, and the triple subject should be ?var
        for i, cond in enumerate(query["where"]["notUnion"]):
            if cond[0] != "Triple":
                continue
            s, p, o = cond[1]
            if (p.split(".")[-1] == "from" or p.split(".")[-1] == "start_date") and not var_from and is_variable(s):
                var_from = o
                index_from = i
            if (p.split(".")[-1] == "to" or p.split(".")[-1] == "end_date") and not var_to and is_variable(s):
                var_to = o
                index_to = i

        if var_from and var_to:

            s_from = query["where"]["notUnion"][index_from][1][0]
            s_to = query["where"]["notUnion"][index_to][1][0]
            p_from = query["where"]["notUnion"][index_from][1][1]
            p_to = query["where"]["notUnion"][index_to][1][1]

            # the subject of from and to are different
            if s_from != s_to:
                flag = False
                for cond in query["where"]["notUnion"]:
                    if cond[0] == "Comparison":
                        s1, op, s2 = cond[1]
                        if (s1 == var_from and s2 == var_to) or (s1 == var_to and s2 == var_from):
                            flag = True
                            break
                assert flag
            # the subject of from and to are the same one
            else:
                num_filter = sum([1 if cond[0] == "Comparison" else 0 for cond in query["where"]["notUnion"]])
                # If the number of filters is less than 2, can not combine from and to
                if depth == 0 and num_filter < 2:
                    return

                var_period = rename("?period", depth)
                if var_from in query["select"] and var_to in query["select"]:
                    query["select"] = [var for var in query["select"] if var != var_from and var != var_to]
                    query["select"].append(var_period)
                query["where"]["notUnion"].pop(max(index_from, index_to))
                query["where"]["notUnion"].pop(min(index_from, index_to))
                query["where"]["notUnion"].append(
                    ["Triple", [s_from, "$$$".join([p_from, p_to.split(".")[-1]]), var_period]])

                assert len(query["where"]["subQueries"]) == 1 or len(query["where"]["subQueries"]) == 0

                # The sub-query do not has filters for combing "from" and "to" to a period.
                if depth > 0:
                    return

                index_from_1, index_to_1 = -1, -1
                sub_from, sub_to, sub_period = None, None, None
                op_from, op_to = None, None
                # Find two filters that related to "from" and "to"
                # such as "FILTER (?to <= ?pto && ?from >= ?pfrom)"
                for i, cond in enumerate(query["where"]["notUnion"]):
                    if cond[0] == "Comparison":
                        s1, op, s2 = cond[1]
                        if s1 == var_from:
                            op_from = op
                            index_from_1 = i
                            sub_from = s2
                        if s1 == var_to:
                            op_to = op
                            index_to_1 = i
                            sub_to = s2
                        if s2 == var_from:
                            op_from = get_inv_op(op)
                            index_from_1 = i
                            sub_from = s1
                        if s2 == var_to:
                            op_to = get_inv_op(op)
                            index_to_1 = i
                            sub_to = s1
                assert index_from_1 != -1 and index_to_1 != -1
                assert (op_from == ">=" and op_to == "<=") or (op_from == "<=" and op_to == ">=") \
                       or (op_from == ">" and op_to == "<") or (op_from == "<" and op_to == ">")
                # Two kinds of overlap for two periods.
                if (op_from == ">=" and op_to == "<=") or (op_from == ">" and op_to == "<"):
                    new_op = "during"
                else:
                    new_op = "overlap"

                query["where"]["notUnion"].pop(max(index_from_1, index_to_1))
                query["where"]["notUnion"].pop(min(index_from_1, index_to_1))

                if len(query["where"]["subQueries"]) == 1:
                    # There is one sub query
                    sub_period = rename("?period", depth + 1)
                else:
                    if "xsd:dateTime" in sub_from:
                        # sub_from and sub_to are both date value.
                        # such as "FILTER(xsd:datetime(?sk1) >= \"2015-01-01\"^^xsd:dateTime) AND
                        #          FILTER(xsd:datetime(?sk3) <= \"2015-12-31\"^^xsd:dateTime)"
                        assert "xsd:dateTime" in sub_to
                        sub_period = "$$$".join([sub_from, sub_to])
                    elif sub_from == sub_to:
                        # sub_from and sub_to are the same variable
                        # such as "FILTER(xsd:datetime(?sk1) >= ?d) AND
                        #          FILTER(xsd:datetime(?sk3) <= ?d)"
                        sub_period = sub_from
                    else:
                        # sub_from and sub_to are two variable in the conditions.
                        # """
                        # SELECT DISTINCT ?x
                        # WHERE {
                        #   ns:m.081pw ns:time.event.start_date ?start ;
                        #              ns:time.event.end_date ?end .
                        #   ?x ns:government.politician.government_positions_held  ?y .
                        #   ?y ns:government.government_position_held.office_position_or_title ns:m.060d2 ; # President of the United States
                        #      ns:government.government_position_held.from  ?from ;
                        #      ns:government.government_position_held.to  ?to .
                        #   FILTER (?from < ?end)
                        #   FILTER (?to > ?start)?x ns:government.politician.government_positions_held ?c .
                        # ?c ns:government.government_position_held.from ?num .
                        # }
                        # ORDER BY ?num LIMIT 1
                        # """
                        # var_from == ?from, var_to == ?to
                        # var_from_2 == ?start, var_to_2 == ?end
                        var_from_2, var_to_2 = None, None
                        index_from_2, index_to_2 = -1, -1
                        for i, cond in enumerate(query["where"]["notUnion"]):
                            if cond[0] != "Triple":
                                continue
                            s, p, o = cond[1]
                            if (p.split(".")[-1] == "from" or p.split(".")[
                                -1] == "start_date") and i != index_from and not var_from_2:
                                var_from_2 = o
                                index_from_2 = i
                            if (p.split(".")[-1] == "to" or p.split(".")[
                                -1] == "end_date") and i != index_to and not var_to_2:
                                var_to_2 = o
                                index_to_2 = i

                        assert var_from_2 and var_to_2
                        # Combine second pair "from"("start") and "to"("end")
                        sub_period = "?second_period_" + str(depth)
                        if var_from_2 in query["select"] and var_to_2 in query["select"]:
                            query["select"] = [var for var in query["select"] if var != var_from_2 and var != var_to_2]
                            query["select"].append(sub_period)
                        s_from_2 = query["where"]["notUnion"][index_from_2][1][0]
                        s_to_2 = query["where"]["notUnion"][index_to_2][1][0]
                        p_from_2 = query["where"]["notUnion"][index_from_2][1][1]
                        p_to_2 = query["where"]["notUnion"][index_to_2][1][1]
                        assert s_from_2 == s_to_2
                        query["where"]["notUnion"].pop(max(index_from_2, index_to_2))
                        query["where"]["notUnion"].pop(min(index_from_2, index_to_2))
                        query["where"]["notUnion"].append(
                            ["Triple", [s_from, "$$$".join([p_from_2, p_to_2.split(".")[-1]]), sub_period]])

                query["where"]["notUnion"].append(["Comparison", [var_period, new_op, sub_period]])


class SPARQLParserCWQ(object):
    def __init__(self):
        self.key_words = ["SELECT", "WHERE", "ORDER", "BY", "AS", "DISTINCT", "LIMIT",
                          "ASC", "DESC", "EXISTS", "NOT", "FILTER", "COUNT", "MAX", "MIN"]

    def preprocess(self, sparql):
        # Remove annotation, begin of "#", end of "\n"
        tmp = ""
        st = 0
        while st < len(sparql):
            if sparql[st] == "#":
                ed = sparql.find("\n", st)
                st = ed
            else:
                tmp += sparql[st]
                st += 1

        sparql = tmp.replace("\n", " ")
        sparql = sparql.replace("\t", " ")
        tokens = sparql.split(" ")
        processed_tokens = []
        for token in tokens:
            if len(token) == 0:
                continue
            new_token = []
            for t in token.split("("):
                new_t = copy.deepcopy(t)
                for key_word in self.key_words:
                    if new_t.lower() == key_word.lower():
                        new_t = key_word
                new_token.append(new_t)
            processed_tokens.append("(".join(new_token))
        processed_sparql = " ".join(processed_tokens)
        processed_sparql = processed_sparql.replace("# ?y2 = country", "")
        return processed_sparql

    def postprocess(self, query, depth, sub_name_mapping):
        # Select one subquery as the main query.
        self.lift_subquery(query, sub_name_mapping)

        # Rename all variable names
        name_mapping = self.rename_variable(query, depth, sub_name_mapping)

        self.simplify_period(query, depth)
        return query, name_mapping

    def parse_sparql(self, sparql, depth):
        sparql = self.preprocess(sparql)
        sparql = sparql.replace("#MANUAL SPARQL", "")

        sparql = sparql.strip(" ")

        select_pos = sparql.find("SELECT")
        where_pos = sparql.find("WHERE")
        # Search from the last "}", that is the end of the WHERE conditions
        order_by_pos = sparql[sparql.rfind("}") + 1 :].find("ORDER BY")
        if order_by_pos != -1:
            order_by_pos += sparql.rfind("}")
        else:
            order_by_pos = len(sparql)

        select_str = sparql[select_pos : where_pos]
        where_str = sparql[where_pos : order_by_pos]
        order_by_str = sparql[order_by_pos:]

        vars, aggs = self.parse_select(select_str)
        conds, sub_name_mapping = self.parse_where(where_str, aggs, depth)
        orders = self.parse_order_by(order_by_str) if len(order_by_str) > 0 else None

        query = {"select": vars,
                 "where": conds,
                 "orderBy": orders}

        query, name_mapping = self.postprocess(query, depth, sub_name_mapping)
        return query, name_mapping

    def parse_variable(self, var_str):
        """
        Parse one variable unit in the SELECT-clause, which may be with an aggregation function or naming statement
        @param var_str: variable string, such as "?x", "MAX(?x)", or "(COUNT(?x) as ?count)"
        @return: aggregation function, variable name, variable alias
        """
        # handle naming statement, e.g., "(COUNT(?x) as ?count)"
        if var_str[0] == "(":
            statement = get_content_from_outermost_brackets(var_str, 0, "(")
            statement = statement.split(" ")
            assert len(statement) == 3 and statement[1] == "AS"
            unit, alias = statement[0], statement[2]
        else:
            unit, alias = var_str, None

        pos = unit.find("(")
        if pos != -1:
            var, l, r = get_content_from_outermost_brackets(unit, pos, "(", return_index=True)
            agg = unit[:l - 1]
        else:
            var = unit
            agg = None
        return agg, var, alias

    def parse_select(self, select_str):
        """
        Parse the SELECT-clause
        @param select_str: such as "SELECT DISTINCT  ?x", "SELECT ?from ?to ?x", "SELECT ?x (COUNT(?x) as ?count)" ...
        @return: is_distinct, list of variables to be selected
        """

        select_str, pos_0 = get_content_behind_key_word(select_str, 0, "SELECT")
        assert pos_0 != -1

        select_str, pos_1 = get_content_behind_key_word(select_str, 0, "DISTINCT")
        distinct = True if pos_1 != -1 else False
        select_str = select_str.strip()

        idx = 0
        select_str_tmp = ""
        while idx < len(select_str):
            s = select_str[idx]
            if s == "(":
                _, l, r = get_content_from_outermost_brackets(select_str, idx, "(", return_index=True)
                # "(COUNT(?x) as ?count)" --> ""(COUNT(?x)_as_?count)" for being split by " "
                select_str_tmp += select_str[l - 1: r + 1].replace(" ", "_")
                idx = r + 1
            else:
                select_str_tmp += s
                idx += 1
        var_list = [var_str for var_str in select_str_tmp.split(" ") if var_str != ""]

        vars = []
        aggs = []
        for var_str in var_list:
            var_str = var_str.replace("_", " ")
            agg, var, alias = self.parse_variable(var_str)
            if agg:
                vars.append(alias)
                aggs.append(["Aggregation", [var, agg, alias]])
            else:
                vars.append(var)
        return vars, aggs

    def parse_where(self, where_str, aggs, depth):
        """
        Parse WHERE-clause
        @param where_str: triples with "UNION", sub-query, and "FILTER"
        @return:
        """
        where_str, pos = get_content_behind_key_word(where_str, 0, "WHERE")
        assert pos != -1

        conds_str = get_content_from_outermost_brackets(where_str, 0, "{")
        conds, name_mapping = self.parse_conditions(conds_str, depth)
        # transform an aggregation function to a triple, such as "COUNT(?x) AS ?count" --> [?x, COUNT, ?count]
        conds["notUnion"] += aggs
        return conds, name_mapping

    def parse_order_by(self, order_by_str):
        # Default initialization
        order = "ASC"
        type = "xsd:integer"

        order_by_str = order_by_str.strip()
        pos_0 = order_by_str.find("ORDER BY")
        pos_1 = order_by_str.find("LIMIT")

        # Variable to be ordered
        var_str = order_by_str[pos_0 + len("ORDER BY"): pos_1].strip()
        ## find order
        if var_str[0:4] == "DESC":
            order = "DESC"
            var_str = var_str[var_str.find("(") + 1: var_str.rfind(")")].strip()
        if var_str[0:3] == "ASC":
            order = "ASC"
            var_str = var_str[var_str.find("(") + 1: var_str.rfind(")")].strip()

        ## find type
        if var_str[0:4] == "xsd:":
            type = var_str[0:var_str.find("(")]
            var_str = var_str[var_str.find("(") + 1: var_str.rfind(")")].strip()
        ## find variable
        var = var_str

        # LIMIT number
        num_str = "\"" + order_by_str[pos_1 + len("LIMIT"):].strip() + "\""
        return [order, type, var, num_str]

    def parse_filter(self, filter_str):
        """
        Parse the content in FILTER()
        @param filter_str:  such as "!isLiteral(?x) OR lang(?x) = '' OR langMatches(lang(?x), 'en')"
                                    "NOT EXISTS {?y ns:government.government_position_held.from ?sk0} ||
                                         EXISTS {?y ns:government.government_position_held.from ?sk1 .
                                                 FILTER(xsd:datetime(?sk1) <= "1980-12-31"^^xsd:dateTime)}"
                                    "NOT EXISTS {?y ns:people.marriage.to []}"
                                    "?x != ns:m.0f8l9c"
                                    "?num < "2007-05-16"^^xsd:dateTime"
                                    "?var >= 65"
        @return:
        """
        # handle "!isLiteral(?x) OR lang(?x) = '' OR langMatches(lang(?x), 'en')"
        if "isLiteral" in filter_str:
            # !isLiteral(?var)
            pos_0 = filter_str.find("!isLiteral(")
            assert pos_0 != -1
            var_0 = get_content_from_outermost_brackets(filter_str, pos_0, "(")

            # lang(?var) = ''
            pos_1 = filter_str.find("lang(")
            assert pos_1 != -1
            var_1 = get_content_from_outermost_brackets(filter_str, pos_1, "(")
            assert var_1 == var_0

            # langMatches(lang(?var), 'en'))
            pos_2 = filter_str.find("langMatches(lang(")
            if pos_2 != -1:
                var_2 = get_content_from_outermost_brackets(filter_str, pos_2 + len("langMatches(") + 1, "(")
            else:
                var_2 = get_content_from_outermost_brackets(filter_str, pos_2 + len("lang") + 1, "(")
            assert var_2 == var_0
            return [["English", var_0]]

        # handle time constraint with "EXISTS"
        elif "EXISTS" in filter_str:
            conds = [cond.strip() for cond in filter_str.split("||")]
            assert len(conds) == 1 or len(conds) == 2

            # For "NOT EXISTS {?y ns:government.government_position_held.from ?sk0} ||
            #      EXISTS {?y ns:government.government_position_held.from ?sk1 . FILTER(xsd:datetime(?sk1) <= "1980-12-31"^^xsd:dateTime) }
            if len(conds) == 2:
                assert "NOT EXISTS" not in conds[0] or "NOT EXISTS" not in conds[1]
                assert ("EXISTS" in conds[0] and "NOT EXISTS" in conds[1]) or ("EXISTS" in conds[1] and "NOT EXISTS" in conds[0])
                if "NOT EXISTS" in conds[1]:
                    conds[0], conds[1] = conds[1], conds[0]

                cond_0 = get_content_from_outermost_brackets(conds[0], conds[0].find("NOT EXISTS"), "{")
                var_0, rel_0, _ = cond_0.strip().split(" ")

                sub_conds = get_content_from_outermost_brackets(conds[1], conds[1].find("EXISTS"), "{")
                cond_1 = sub_conds[:sub_conds.find(" . ")]
                var_1, rel_1, var_2 = cond_1.strip().split(" ")
                assert var_0 == var_1
                assert rel_0 == rel_1
                assert "FILTER" in sub_conds

                # Time constraint
                sub_filter_str = get_content_from_outermost_brackets(sub_conds, sub_conds.find("FILTER"), "(")
                sub_constraints = self.parse_filter(sub_filter_str)
                return sub_constraints + [["Triple", [var_0, rel_0, var_2]]]

            # FILTER (NOT EXISTS {?y ns:people.marriage.to []})
            else:
                cond = get_content_from_outermost_brackets(conds[0], conds[0].find("EXISTS"), "{")
                return [["Not Exists", cond.strip().split(" ")]]

        # handle Comparison condition
        # such as "?x != ns:m.0f8l9c", "?num < "2007-05-16"^^xsd:dateTime", "?var >= 65"
        elif "=" in filter_str or ">" in filter_str or "<" in filter_str:
            assert "||" not in filter_str or "&&" not in filter_str
            join_op = "||" if "||" in filter_str else "&&"
            conds = [x.strip() for x in filter_str.split(join_op)]
            if join_op == "||":      # "?k = ns:m.04627hw || ?k = ns:m.04627gn"
                conds = conds[:1]

            OPS = ["=", "!=", ">", ">=", "<", "<="]
            constraints = []
            for cond in conds:
                constraint = [var.strip() for var in cond.split(" ") if var != ""]
                mid = -1
                for i, cons in enumerate(constraint):
                    if cons in OPS:
                        mid = i
                        break
                assert mid != -1

                if mid == 3:
                    # ['xsd:dateTime(?pFrom)', '-', 'xsd:dateTime(?from)', '>', '0']
                    assert constraint[1] == "-" and constraint[-1] == "0"
                    constraints.append(["Comparison", [remove_type(constraint[0]), constraint[3], remove_type(constraint[2])]])
                elif mid == 1 and len(constraint) - mid > 2:
                    # ['str(?sk0)', '=', '"Young', 'Forrest"']
                    assert "str" in constraint[0] and constraint[-1][-1] == "\""
                    constraints.append(["Comparison", [remove_type(constraint[0]), constraint[1], " ".join(constraint[2:])]])
                else:
                    constraints.append(["Comparison", [remove_type(constraint[0]), constraint[1], remove_type(constraint[2])]])
            return constraints

        else:
            raise ValueError("There is another type of FILTER, \"{}\".".format(filter_str))

    def parse_conditions(self, conds_str, depth):
        """
        Parse conditions in WHERE-clause
        @param conds_str: including triples, or {...}, or "FILTER"
                          {...} can be joined by "UNION", and may also includes the sub-query.
        @return:
        """
        name_mapping = {}
        conds = {"notUnion": [], "union": [], "subQueries": []}
        # handle sub query
        # SELECT (MAX(?count) AS ?maxCount)
        # WHERE {
        #     SELECT ?x (COUNT(?x) as ?count)
        #     WHERE {
        #       ns:m.02_p0 ns:sports.sports_award_type.winners ?y .
        #       ?y ns:sports.sports_award.award_winner ?x .
        #     }
        # }
        if get_content_behind_key_word(conds_str, 0, "SELECT", only_index=True) != -1:
            sub_query, sub_name_mapping = self.parse_sparql(conds_str, depth=depth + 1)
            conds["subQueries"].append(sub_query)
            return conds, sub_name_mapping

        pos = 0
        while pos < len(conds_str):

            # Handle {...}:
            if conds_str[pos] == "{":
                cond_str, l_idx, r_idx = get_content_from_outermost_brackets(conds_str, pos, "{", return_index=True)
                cond_str = cond_str.strip()
                if "SELECT" in cond_str:
                    # handle sub query
                    # {
                    #     SELECT ?pfrom ?pto
                    #     WHERE {
                    #       ns:m.083q7 ns:government.politician.government_positions_held ?y0 .  # Woodrow Wilson
                    #       ?y0 ns:government.government_position_held.basic_title ns:m.060c4 ;  # President
                    #           ns:government.government_position_held.from ?pfrom ;
                    #           ns:government.government_position_held.to ?pto .
                    #     }
                    #  }
                    sub_query, sub_name_mapping = self.parse_sparql(cond_str, depth=depth + 1)
                    name_mapping = combine_mapping(name_mapping, sub_name_mapping)
                    conds["subQueries"].append(sub_query)
                else:
                    # """
                    # {
                    #    ns:m.05r4w ns:location.statistical_region.places_exported_to ?y .
                    #    ?y ns:location.imports_and_exports.exported_to ?x .
                    #  }
                    # """
                    sub_conds, sub_name_mapping = self.parse_conditions(cond_str, depth=depth)
                    name_mapping = combine_mapping(name_mapping, sub_name_mapping)
                    pos_tmp = get_content_behind_key_word(conds_str, r_idx + 1, "UNION", only_index=True)
                    if pos_tmp == -1:
                        conds["notUnion"] += sub_conds["notUnion"]
                        conds["union"] += sub_conds["union"]
                    else:
                        assert len(sub_conds["union"]) == 0 and len(sub_conds["subQueries"]) == 0
                        conds["union"].append(sub_conds["notUnion"])

                    while pos_tmp != -1:
                        # find following UNION {...}
                        # """
                        #  UNION
                        #  {
                        #     ns:m.05r4w ns:location.statistical_region.places_imported_from ?y .
                        #     ?y ns:location.imports_and_exports.imported_from ?x .
                        #  }
                        # """
                        cond_str, l_idx, r_idx = get_content_from_outermost_brackets(conds_str, pos_tmp, "{", return_index=True)
                        pos_tmp = get_content_behind_key_word(conds_str, r_idx + 1, "UNION", only_index=True)
                        sub_conds, sub_name_mapping = self.parse_conditions(cond_str, depth=depth)
                        name_mapping = combine_mapping(name_mapping, sub_name_mapping)

                        assert len(sub_conds["union"]) == 0 and len(sub_conds["subQueries"]) == 0
                        conds["union"].append(sub_conds["notUnion"])

                pos = r_idx + 1
                continue

            # Handle Filter(...):
            pos_tmp = get_content_behind_key_word(conds_str, pos, "FILTER", only_index=True)
            if pos_tmp != -1:
                cond_str, l_idx, r_idx = get_content_from_outermost_brackets(conds_str, pos_tmp, "(", return_index=True)
                constraints = self.parse_filter(cond_str)
                conds["notUnion"] += constraints
                pos = r_idx + 1
                continue

            # Find one triple condition, start from "?" or "ns:", end in " . "
            if conds_str[pos] == "?" or conds_str[pos: min(pos + len("ns:"), len(conds_str))] == "ns:":
                suffix = " ."
                l_idx = pos
                r_idx = conds_str.find(" .", pos)
                if r_idx == -1:
                    r_idx = len(conds_str)
                one_cond_str = conds_str[l_idx: r_idx]
                sub_conds = self.parse_triples(one_cond_str)

                conds["notUnion"].extend(sub_conds)
                pos = r_idx + len(suffix)
                continue
            pos += 1
        return conds, name_mapping

    def parse_triples(self, one_cond_str):
        # ?y0 ns:government.government_position_held.basic_title ns:m.060c4 ;  # President
        #     ns:government.government_position_held.from ?pfrom ;
        #     ns:government.government_position_held.to ?pto .
        units = []
        i = 0
        conds = [x for x in one_cond_str.split(" ") if len(x) > 0]
        while i < len(conds):
            unit = conds[i]
            if unit == ";" or unit[0] == "?" or (unit[0] == "\"" and "\"" in unit[1:]) or \
                    unit[0] == "<" or unit[:min(len("ns:"), len(unit))] == "ns:":
                units.append(unit)
                i += 1
            elif unit[0] == "\"":
                j = i + 1
                while j < len(conds) and "\"" not in conds[j]:
                    j += 1
                units.extend(conds[i:j + 1])
                i = j + 1
            else:
                i += 1

        # for i, unit in enumerate(one_cond_str.split(" ")):
        #     if len(unit) == 0: continue
        #     if unit == ";" or unit[0] == "?" or unit[0] == "\"" or \
        #             unit[0] == "<" or unit[:min(len("ns:"), len(unit))] == "ns:":
        #         units.append(unit)
        units = " ".join(units).split(" ; ")
        s, p, o = self.parse_one_triple(units[0])
        triples = [["Triple", [s, p, o]]]
        for unit in units[1:]:
            p, o = self.parse_one_triple(unit.strip())
            triples.append(["Triple", [s, p, o]])
        return triples

    def parse_one_triple(self, unit):
        triple = unit.split(" ")
        if len(triple) == 3:
            assert "ns:" in triple[0] or "?" in triple[0]
            assert "ns:" in triple[1]
            return triple[0], triple[1], triple[2]
        elif len(triple) == 2:
            assert "ns:" in triple[0]
            return triple[0], triple[1]
        elif len(triple) > 3:
            assert "ns:" in triple[0] or "?" in triple[0]
            assert "ns:" in triple[1]
            assert "\"" in triple[2]
            assert "\"" in triple[-1]
            return triple[0], triple[1], " ".join(triple[2:])
        else:
            raise ValueError("Triple \"{}\" is not correct !".format(unit))

    def lift_subquery(self, query, sub_name_mapping):
        """
        when main query do not has triples, choose one sub query as the main query.

        SELECT DISTINCT  ?x
         WHERE
         {
           {
             SELECT ?pfrom ?pto
             WHERE {
               ns:m.083q7 ns:government.politician.government_positions_held ?y0 .   # Woodrow Wilson
               ?y0 ns:government.government_position_held.basic_title ns:m.060c4 ;   # President
                   ns:government.government_position_held.from ?pfrom ;
                   ns:government.government_position_held.to ?pto .
             }
           }
           {
             SELECT ?from ?to ?x
             WHERE {
               ?x ns:common.topic.notable_types ns:m.02h76fz .  # Military Conflict
               ?x ns:time.event.start_date ?from .
               ?x ns:time.event.end_date ?to .
               ?x ns:military.military_conflict.combatants ?y .
               ?y ns:military.military_combatant_group.combatants ns:m.09c7w0 .   # United States of America
             }
           }
           FILTER (?to <= ?pto && ?from >= ?pfrom)
         }
         ==================================>
         SELECT DISTINCT  ?x
         WHERE
         {
           {
             SELECT ?pfrom ?pto
             WHERE {
               ns:m.083q7 ns:government.politician.government_positions_held ?y0 .  # Woodrow Wilson
               ?y0 ns:government.government_position_held.basic_title ns:m.060c4 ;  # President
                   ns:government.government_position_held.from ?pfrom ;
                   ns:government.government_position_held.to ?pto .
             }
           }
           ?x ns:common.topic.notable_types ns:m.02h76fz .  Military Conflict
           ?x ns:time.event.start_date ?from .
           ?x ns:time.event.end_date ?to .
           ?x ns:military.military_conflict.combatants ?y .
           ?y ns:military.military_combatant_group.combatants ns:m.09c7w0 .   United States of America
           FILTER (?to <= ?pto && ?from >= ?pfrom)
         }
        
        @param query: 
        @param sub_name_mapping: 
        @return: 
        """
        index = -1
        for i, sub_query in enumerate(query["where"]["subQueries"]):
            if len(sub_query["select"]) > 1 and "?x" in [rename_inv("?x") for x in sub_query["select"]]:
                index = i
                break
        if index != -1:
            sub_query = copy.deepcopy(query["where"]["subQueries"][index])
            query["where"]["subQueries"].pop(index)

            sub_query_not_union = []
            for type, cond in sub_query["where"]["notUnion"]:
                if type == "English":
                    cond = rename_inv(cond)
                    sub_query_not_union.append([type, cond])
                    if cond in sub_name_mapping:
                        sub_name_mapping.pop(cond)
                else:
                    s, p, o = cond
                    if is_variable(s) or is_entity(s):
                        s = rename_inv(s)
                        if s in sub_name_mapping:
                            sub_name_mapping.pop(s)
                    if is_variable(o) or is_entity(o):
                        o = rename_inv(o)
                        if o in sub_name_mapping:
                            sub_name_mapping.pop(o)
                    sub_query_not_union.append([type, [s, p, o]])

            index_period = -1
            p_period = None
            s_period = None
            for i, cond in enumerate(sub_query_not_union):
                if cond[0] == "Triple" and "period" in cond[1][2]:
                    index_period = i
                    s_period = cond[1][0]
                    p_period = cond[1][1]

            if index_period != -1:
                p_from, p_to = p_period.split("$$$")
                sub_query_not_union.pop(index_period)
                sub_query_not_union.append(["Triple", [s_period, p_from, "?from"]])
                sub_query_not_union.append(["Triple", [s_period, p_to, "?to"]])

            sub_query_union = []
            for conds in sub_query["where"]["union"]:
                new_conds = []
                for type, cond in conds:
                    if type == "English":
                        cond = rename_inv(cond)
                        new_conds.append([type, cond])
                        if cond in sub_name_mapping:
                            sub_name_mapping.pop(cond)
                    else:
                        s, p, o = cond
                        if is_variable(s):
                            s = rename_inv(s)
                            if s in sub_name_mapping:
                                sub_name_mapping.pop(s)
                        if is_variable(o):
                            o = rename_inv(o)
                            if o in sub_name_mapping:
                                sub_name_mapping.pop(o)
                        new_conds.append([type, [s, p, o]])
                sub_query_union.append(new_conds)

            sub_query_order_by = copy.deepcopy(sub_query["orderBy"])
            if sub_query_order_by:
                sub_query_order_by[2] = rename_inv(sub_query_order_by[2])
                sub_name_mapping.pop(sub_query_order_by[2])

            query["where"]["notUnion"] = sub_query_not_union + query["where"]["notUnion"]
            query["where"]["union"] = sub_query_union + query["where"]["union"]

            assert not query["orderBy"] or not sub_query_order_by
            if not query["orderBy"]:
                query["orderBy"] = sub_query_order_by

            assert len(sub_query["where"]["subQueries"]) == 0

    def simplify_period(self, query, depth):
        """
        In the time constraint, combine "from" relation AND "to" relation into a new "period" relation.
        Avoid there is circle in the query graph

        ?y ns:government.government_position_held.from ?pfrom ;
        ?y ns:government.government_position_held.to ?pto .
        ?x ns:time.event.start_date ?from .
        ?x ns:time.event.end_date ?to .
        FILTER (?to <= ?pto && ?from >= ?pfrom)
        =============================================>
        ?y ns:government.government_position_held.from###ns:government.government_position_held.to ?sub_period .
        ?x ns:time.event.start_date###ns:time.event.end_date ?period .
        FILTER (?period during ?sub_period)

        @param query: parsed query 
        @param depth: main query(0), sub query (1) ...
        """
        var_from, var_to = None, None
        index_from, index_to = -1, -1
        # Time constraint must not be union
        # Find the first ?from and ?to, and the triple subject should be ?var
        for i, cond in enumerate(query["where"]["notUnion"]):
            if cond[0] != "Triple":
                continue
            s, p, o = cond[1]
            if (p.split(".")[-1] == "from" or p.split(".")[-1] == "start_date") and not var_from and is_variable(s):
                var_from = o
                index_from = i
            if (p.split(".")[-1] == "to" or p.split(".")[-1] == "end_date") and not var_to and is_variable(s):
                var_to = o
                index_to = i

        if var_from and var_to:

            s_from = query["where"]["notUnion"][index_from][1][0]
            s_to = query["where"]["notUnion"][index_to][1][0]
            p_from = query["where"]["notUnion"][index_from][1][1]
            p_to = query["where"]["notUnion"][index_to][1][1]

            # the subject of from and to are different
            if s_from != s_to:
                flag = False
                for cond in query["where"]["notUnion"]:
                    if cond[0] == "Comparison":
                        s1, op, s2 = cond[1]
                        if (s1 == var_from and s2 == var_to) or (s1 == var_to and s2 == var_from):
                            flag = True
                            break
                assert flag
            # the subject of from and to are the same one
            else:
                num_filter = sum([1 if cond[0] == "Comparison" else 0 for cond in query["where"]["notUnion"]])
                # If the number of filters is less than 2, can not combine from and to
                if depth == 0 and num_filter < 2:
                    return

                var_period = rename("?period", depth)
                if var_from in query["select"] and var_to in query["select"]:
                    query["select"] = [var for var in query["select"] if var != var_from and var != var_to]
                    query["select"].append(var_period)
                query["where"]["notUnion"].pop(max(index_from, index_to))
                query["where"]["notUnion"].pop(min(index_from, index_to))
                query["where"]["notUnion"].append(["Triple", [s_from, "$$$".join([p_from, p_to.split(".")[-1]]), var_period]])

                assert len(query["where"]["subQueries"]) == 1 or len(query["where"]["subQueries"]) == 0

                # The sub-query do not has filters for combing "from" and "to" to a period.
                if depth > 0:
                    return

                index_from_1, index_to_1 = -1, -1
                sub_from, sub_to, sub_period = None, None, None
                op_from, op_to = None, None
                # Find two filters that related to "from" and "to"
                # such as "FILTER (?to <= ?pto && ?from >= ?pfrom)"
                for i, cond in enumerate(query["where"]["notUnion"]):
                    if cond[0] == "Comparison":
                        s1, op, s2 = cond[1]
                        if s1 == var_from:
                            op_from = op
                            index_from_1 = i
                            sub_from = s2
                        if s1 == var_to:
                            op_to = op
                            index_to_1 = i
                            sub_to = s2
                        if s2 == var_from:
                            op_from = get_inv_op(op)
                            index_from_1 = i
                            sub_from = s1
                        if s2 == var_to:
                            op_to = get_inv_op(op)
                            index_to_1 = i
                            sub_to = s1
                assert index_from_1 != -1 and index_to_1 != -1
                assert (op_from == ">=" and op_to == "<=") or (op_from == "<=" and op_to == ">=") \
                       or (op_from == ">" and op_to == "<") or (op_from == "<" and op_to == ">")
                # Two kinds of overlap for two periods.
                if (op_from == ">=" and op_to == "<=") or (op_from == ">" and op_to == "<"):
                    new_op = "during"
                else:
                    new_op = "overlap"

                query["where"]["notUnion"].pop(max(index_from_1, index_to_1))
                query["where"]["notUnion"].pop(min(index_from_1, index_to_1))

                if len(query["where"]["subQueries"]) == 1:
                    # There is one sub query
                    sub_period = rename("?period", depth + 1)
                else:
                    if "xsd:dateTime" in sub_from:
                        # sub_from and sub_to are both date value.
                        # such as "FILTER(xsd:datetime(?sk1) >= \"2015-01-01\"^^xsd:dateTime) AND
                        #          FILTER(xsd:datetime(?sk3) <= \"2015-12-31\"^^xsd:dateTime)"
                        assert "xsd:dateTime" in sub_to
                        sub_period = "$$$".join([sub_from, sub_to])
                    elif sub_from == sub_to:
                        # sub_from and sub_to are the same variable
                        # such as "FILTER(xsd:datetime(?sk1) >= ?d) AND
                        #          FILTER(xsd:datetime(?sk3) <= ?d)"
                        sub_period = sub_from
                    else:
                        # sub_from and sub_to are two variable in the conditions.
                        # """
                        # SELECT DISTINCT ?x
                        # WHERE {
                        #   ns:m.081pw ns:time.event.start_date ?start ;
                        #              ns:time.event.end_date ?end .
                        #   ?x ns:government.politician.government_positions_held  ?y .
                        #   ?y ns:government.government_position_held.office_position_or_title ns:m.060d2 ; # President of the United States
                        #      ns:government.government_position_held.from  ?from ;
                        #      ns:government.government_position_held.to  ?to .
                        #   FILTER (?from < ?end)
                        #   FILTER (?to > ?start)?x ns:government.politician.government_positions_held ?c .
                        # ?c ns:government.government_position_held.from ?num .
                        # }
                        # ORDER BY ?num LIMIT 1
                        # """
                        # var_from == ?from, var_to == ?to
                        # var_from_2 == ?start, var_to_2 == ?end
                        var_from_2, var_to_2 = None, None
                        index_from_2, index_to_2 = -1, -1
                        for i, cond in enumerate(query["where"]["notUnion"]):
                            if cond[0] != "Triple":
                                continue
                            s, p, o = cond[1]
                            if (p.split(".")[-1] == "from" or p.split(".")[-1] == "start_date") and i != index_from and not var_from_2:
                                var_from_2 = o
                                index_from_2 = i
                            if (p.split(".")[-1] == "to" or p.split(".")[-1] == "end_date") and i != index_to and not var_to_2:
                                var_to_2 = o
                                index_to_2 = i

                        assert var_from_2 and var_to_2
                        # Combine second pair "from"("start") and "to"("end")
                        sub_period = "?second_period_" + str(depth)
                        if var_from_2 in query["select"] and var_to_2 in query["select"]:
                            query["select"] = [var for var in query["select"] if var != var_from_2 and var != var_to_2]
                            query["select"].append(sub_period)
                        s_from_2 = query["where"]["notUnion"][index_from_2][1][0]
                        s_to_2 = query["where"]["notUnion"][index_to_2][1][0]
                        p_from_2 = query["where"]["notUnion"][index_from_2][1][1]
                        p_to_2 = query["where"]["notUnion"][index_to_2][1][1]
                        assert s_from_2 == s_to_2
                        query["where"]["notUnion"].pop(max(index_from_2, index_to_2))
                        query["where"]["notUnion"].pop(min(index_from_2, index_to_2))
                        query["where"]["notUnion"].append(["Triple", [s_from, "$$$".join([p_from_2, p_to_2.split(".")[-1]]), sub_period]])

                # WebQTest-178_354fb646be304159441f2497ab0a9c62
                # WebQTest-1_2e5a22514a269da41a0f16afab223478
                # WebQTest-813_f57701d4be9b1b3ad7b426121ad27b5c
                # WebQTrn-3551_c4ca6374be459506d19081c73420da07
                # WebQTrn-2439_948a1175c630b25f521ec04c20000fd9
                # WebQTrn-2707_66905cd11ca592b6417317c6f01db6f6
                # WebQTest-178_77dde5a026e1fc4a375d8f7326bd62f8
                query["where"]["notUnion"].append(["Comparison", [var_period, new_op, sub_period]])

    def rename_variable(self, query, depth, sub_name_mapping):
        """
        Rename all variables in the sparql by their depth (whether in sub-query), ONLY HANDLE THE QUERY OF CURRENT DEPTH
        @param query: processed query (dict)
        @param depth: main query(0), sub query (1) ...
        @param sub_name_mapping: name mapping of the sub query.
        @return: name mapping of all variables (including name mapping in the sub query) 
        """
        # make name mapping by the triples in WHERE
        name_mapping = {}  # save name mapping.
        for i in range(len(query["where"]["notUnion"])):
            if query["where"]["notUnion"][i][0] != "Triple" and query["where"]["notUnion"][i][0] != "Aggregation":
                continue
            for j in [0, 2]:  # subject or object
                if is_variable(query["where"]["notUnion"][i][1][j]) or is_entity(query["where"]["notUnion"][i][1][j]):
                    origin_name = copy.deepcopy(query["where"]["notUnion"][i][1][j])
                    if origin_name not in name_mapping:
                        name_mapping[origin_name] = rename(origin_name, depth)
                    query["where"]["notUnion"][i][1][j] = name_mapping[origin_name]

        for i in range(len(query["where"]["union"])):
            for k in range(len(query["where"]["union"][i])):
                if query["where"]["union"][i][k][0] != "Triple" and query["where"]["union"][i][k][0] != "Aggregation":
                    continue
                for j in [0, 2]:  # subject or object
                    if is_variable(query["where"]["union"][i][k][1][j]) or is_entity(query["where"]["union"][i][k][1][j]):
                        origin_name = copy.deepcopy(query["where"]["union"][i][k][1][j])
                        if origin_name not in name_mapping:
                            name_mapping[origin_name] = rename(origin_name, depth)
                        query["where"]["union"][i][k][1][j] = name_mapping[origin_name]

        # combine name mapping in the current query and sub query.
        name_mapping = combine_mapping(name_mapping, sub_name_mapping)

        # rename SELECT
        for i in range(len(query["select"])):
            query["select"][i] = name_mapping[query["select"][i]]

        # rename FILTER
        for i in range(len(query["where"]["notUnion"])):
            if query["where"]["notUnion"][i][0] == "English":
                query["where"]["notUnion"][i][1] = name_mapping[query["where"]["notUnion"][i][1]]
            if query["where"]["notUnion"][i][0] == "Comparison":
                for j in [0, 2]:  # subject or object
                    if is_variable(query["where"]["notUnion"][i][1][j]):
                        query["where"]["notUnion"][i][1][j] = name_mapping[query["where"]["notUnion"][i][1][j]]
                    elif is_entity(query["where"]["notUnion"][i][1][j]):
                        query["where"]["notUnion"][i][1][j] = rename(query["where"]["notUnion"][i][1][j], depth)

        for i in range(len(query["where"]["union"])):
            for k in range(len(query["where"]["union"][i])):
                if query["where"]["union"][i][k][0] == "English":
                    query["where"]["union"][i][k][1] = name_mapping[query["where"]["union"][i][k][1]]
                if query["where"]["union"][i][k][0] == "Comparison":
                    for j in [0, 2]:  # subject or object
                        if is_variable(query["where"]["union"][i][k][1][j]):
                            query["where"]["union"][i][k][1][j] = name_mapping[query["where"]["union"][i][k][1][j]]
                        elif is_entity(query["where"]["notUnion"][i][1][j]):
                            query["where"]["notUnion"][i][k][1][j] = rename(query["where"]["notUnion"][i][k][1][j], depth)

        # rename order by
        if query["orderBy"]:
            query["orderBy"][2] = name_mapping[query["orderBy"][2]]

        return name_mapping


class Annotator():
    """
    an annotator for transform query to the ground truth of AQG operation labels.
    """

    def __init__(self, dataset):
        self.dataset = dataset
        if dataset == "lcq":
            self.kb = "dbpedia"
        else:
            self.kb = "freebase"

        self.v_classes = {v: k for k, v in V_CLASS_IDS.items()}
        self.e_classes = {v: k for k, v in E_CLASS_IDS.items()}
        self.stop_signal = len(self.v_classes) - 1

    def get_v_class(self, v, start_v="?x"):
        if v == start_v:
            return V_CLASS_IDS["ans"]  # answer

        if is_variable(v):
            return V_CLASS_IDS["var"]  # variable

        if is_entity(v, kb=self.kb):
            return V_CLASS_IDS["ent"]  # entity

        if is_type(v, kb=self.kb):
            return V_CLASS_IDS["type"]  # type

        if is_value(v):
            # \"1934\"^^xsd:dateTime
            # 2
            # -799223
            # <http://youtu.be/0bdZWrW6HnA>
            return V_CLASS_IDS["val"]  # value

        raise ValueError('Wrong vertex type: {}'.format(v))

    def get_e_class(self, e, inv=False):
        if e in AGG_NAME:
            if not inv:
                return E_CLASS_IDS["agg+"]  # Aggregation
            else:
                return E_CLASS_IDS["agg-"]  # Aggregation

        if e in CMP_NAME:
            if not inv:
                return E_CLASS_IDS["cmp+"]  # Comparison
            else:
                return E_CLASS_IDS["cmp-"]  # Comparison

        if e in ORD_NAME:
            if not inv:
                return E_CLASS_IDS["ord+"]  # Order
            else:
                return E_CLASS_IDS["ord-"]  # Order

        if is_relation(e, kb=self.kb):
            if not inv:
                return E_CLASS_IDS["rel+"]  # direction "+" Rel
            else:
                return E_CLASS_IDS["rel-"]  # direction "-" Rel

        raise ValueError('Wrong edge type: {}'.format(e))

    def annotate_query(self, query, instance_pool, kb, kb_endpoint, training=False):

        assert len(query["select"]) == 1
        assert len(query["where"]["subQueries"]) == 0 or len(query["where"]["subQueries"]) == 1
        if len(query["where"]["subQueries"]) == 1:
            assert len(query["where"]["subQueries"][0]["where"]["union"]) == 0

        aqg_list = []
        aqg_obj_labels_list = []
        v_instance_obj_labels_list = []
        e_instance_obj_labels_list = []
        v_copy_labels_list = []
        e_copy_labels_list = []
        segment_switch_labels_list = []
        instance_pool_with_gold_list = []

        start_v_name = query["select"][0]

        if len(query["where"]["union"]) == 0:
            query["where"]["union"].append([])

        # The content in each union serves as an independent sample
        for union_conds in query["where"]["union"]:

            # Get all vertices names and edge triples
            # vertices: {v_name}
            # edges: {[v1, e, v2]}
            vertices, edges = self.extract_vertices_and_edges(union_conds + query["where"]["notUnion"], query["orderBy"])
            v_layers = {v: 0 for i, v in enumerate(vertices)}

            if len(query["where"]["subQueries"]) == 1:
                sub_query = query["where"]["subQueries"][0]
                assert len(sub_query["select"]) == 1

                # sub_vertices: {v_name}
                # sub_edges: {[v1, e, v2]}
                sub_vertices, sub_edges = self.extract_vertices_and_edges(sub_query["where"]["notUnion"], sub_query["orderBy"])
                sub_v_layers = {v: 1 for i, v in enumerate(sub_vertices)}

                # Combining, cover the layer for the connected vertex, such as "?f1"
                v_layers = combine_mapping(v_layers, sub_v_layers, cover=True)
                vertices = list(set(vertices + sub_vertices))
                edges += sub_edges

            vertices.sort()
            edges.sort()

            # random.shuffle(vertices)
            # random.shuffle(edges)

            if len(vertices) != len(edges) + 1:
                return [], [], [], [], [], [], [], []

            # build the gold AQG
            aqg = AbstractQueryGraph()

            # add vertex
            v_ids = dict()
            for v_name in vertices:
                if v_name not in v_ids:
                    # attach ID to each vertex as the unique identifier
                    v_ids[v_name] = len(v_ids)
                v_i = v_ids[v_name]
                v_class = self.get_v_class(v_name, start_v=start_v_name)

                aqg.add_vertex(v_i, v_class)
                aqg.set_vertex_segment(v_i, v_layers[v_name])     # set the subquery (segment) id of v
                if v_class != V_CLASS_IDS["var"] and v_class != V_CLASS_IDS["ans"]:
                    aqg.set_vertex_instance(v_i, [-1, rename_inv(v_name)])

            # add edge
            # Sort all edges and take the index as the unique identifier of the edges
            # It can also attach the same index to the edge with the same name.
            e_ids = dict()
            for v_name1, e_name, v_name2 in edges:
                e_name_inv = get_inv_name(e_name)

                if e_name not in e_ids:
                    e_ids[e_name] = len(e_ids)
                    e_ids[e_name_inv] = len(e_ids)

                v_i1 = v_ids[v_name1]
                v_i2 = v_ids[v_name2]
                v_layer1 = aqg.get_vertex_segment(v_i1)
                v_layer2 = aqg.get_vertex_segment(v_i2)

                e_i = e_ids[e_name]
                e_i_inv = e_ids[e_name_inv]
                e_class = self.get_e_class(e_name)
                e_class_inv = self.get_e_class(e_name, inv=True)

                # v1 --> v2
                aqg.add_edge(e_i, e_class, v_i1, v_i2, both_ends=False)
                aqg.set_edge_segment(e_i, max(v_layer1, v_layer2))
                aqg.set_edge_instance(e_i, [-1, e_name])

                # v2 --> v1, add it for transformer and traversal
                aqg.add_edge(e_i_inv, e_class_inv, v_i2, v_i1, both_ends=False)
                aqg.set_edge_segment(e_i_inv, max(v_layer1, v_layer2))
                aqg.set_edge_instance(e_i_inv, [-1, e_name])

            instance_pool_with_gold = build_instance_pool_with_gold(instance_pool, aqg, kb, kb_endpoint)

            v_start = v_ids[start_v_name]

            # build labels of the main query
            aqg_obj_labels, v_instance_obj_labels, e_instance_obj_labels, \
            v_copy_labels, e_copy_labels, segment_switch_labels, new_v_ids, new_e_ids = self.build_labels(aqg,
                                                                                                          layer=0,
                                                                                                          v_start=v_start,
                                                                                                          instance_pool_with_gold=instance_pool_with_gold,
                                                                                                          training=training)

            if len(query["where"]["subQueries"]) == 1:
                # build labels of the sub-query
                sub_aqg_obj_labels, sub_v_instance_obj_labels, sub_e_instance_obj_labels, \
                sub_v_copy_labels, sub_e_copy_labels, sub_segment_switch_labels, _, _ = self.build_labels(aqg,
                                                                                                          layer=1,
                                                                                                          v_start=v_start,
                                                                                                          instance_pool_with_gold=instance_pool_with_gold,
                                                                                                          v_ids=new_v_ids,
                                                                                                          e_ids=new_e_ids,
                                                                                                          training=training)
                aqg_obj_labels += sub_aqg_obj_labels
                v_instance_obj_labels += sub_v_instance_obj_labels
                e_instance_obj_labels += sub_e_instance_obj_labels
                v_copy_labels += sub_v_copy_labels
                e_copy_labels += sub_e_copy_labels
                segment_switch_labels += sub_segment_switch_labels

            assert len(aqg_obj_labels) == 3 * len(e_copy_labels) + 1
            assert len(aqg_obj_labels) == 3 * len(segment_switch_labels) - 2
            assert len(v_copy_labels) == len(segment_switch_labels)

            # terminal signal
            aqg_obj_labels.append(self.stop_signal)

            aqg_list.append(aqg)
            aqg_obj_labels_list.append(aqg_obj_labels)
            v_instance_obj_labels_list.append(v_instance_obj_labels)
            e_instance_obj_labels_list.append(e_instance_obj_labels)
            v_copy_labels_list.append(v_copy_labels)
            e_copy_labels_list.append(e_copy_labels)
            segment_switch_labels_list.append(segment_switch_labels)
            instance_pool_with_gold_list.append(instance_pool_with_gold)

        return aqg_list, aqg_obj_labels_list, \
               v_instance_obj_labels_list, e_instance_obj_labels_list, \
               v_copy_labels_list, e_copy_labels_list, \
               segment_switch_labels_list, \
               instance_pool_with_gold_list

    def extract_vertices_and_edges(self, conds, order_by):
        edges = []
        # not union triples
        for type, cond in conds:
            if type == "English" or type == "Not Exists":
                continue
            if type == "Comparison" and cond[1] == "!=":
                continue
            assert type == "Triple" or type == "Aggregation" or type == "Comparison"
            edges.append(cond)

        if order_by:
            order, _, var, number = order_by
            edges.append([var, order, number])

        edges = simplify_variable_in_filter(edges)

        vertices = list(set(sum(([t[0], t[2]] for t in edges), [])))
        vertices.sort()
        return vertices, edges

    def build_labels(self, aqg, layer, v_start, instance_pool_with_gold, v_ids=None, e_ids=None, training=False):
        """
        build ground-truth labels for generation process
        @param aqg:                         gold abstract query graph
        @param layer:                       current subquery layer (segment) for building labels
        @param instance_pool_with_gold:     DICT: {"vertex", {"ent": {...}, "val": {...} ... },
                                                   "edge", {"rel+": {...}, "agg": {...} ... } }
        @param v_ids:                       DICT: the travel index of the vertex
        @param e_ids:                       DICT: the travel index of the edge
        @return:
        """
        assert (not v_ids and not e_ids) or (v_ids and e_ids)

        def get_instance_id_label(name, instance_list):
            label = -1
            for i, instance in enumerate(instance_list):
                if instance[0] == name:
                    label = i
            assert label != -1
            return label

        def dfs(current_vertex, layer):
            for next_vertex, edge in aqg.neighbours[current_vertex]:
                next_layer = aqg.get_vertex_segment(next_vertex)
                # only travel the layers which higher or equal with current layer
                if next_layer <= layer and next_vertex not in visit:
                    visit.add(next_vertex)
                    # only build label for current layer.
                    if next_layer == layer:
                        next_v_class = aqg.get_vertex_label(next_vertex)
                        next_e_class = aqg.get_edge_label(edge)

                        ####### Build the copy labels for vertex
                        # Copy vertex only when there are two same entities in the main-query and sub-query.
                        cp_v_label = -1
                        if next_v_class == V_CLASS_IDS["ent"]:
                            next_v_name = aqg.get_vertex_instance(next_vertex)[-1]
                            for v, v_id in v_ids.items():
                                if aqg.get_vertex_label(v) == V_CLASS_IDS["ans"] or aqg.get_vertex_label(v) == V_CLASS_IDS["var"]:
                                    continue
                                if rename_inv(next_v_name) in aqg.get_vertex_instance(v)[-1]:
                                    cp_v_label = v_id
                                    break
                        v_copy_labels.append(cp_v_label)

                        ####### Build the copy labels for edges
                        cp_e_label = -1
                        if edge in e_ids:
                            cp_e_label = e_ids[edge]
                        e_copy_labels.append(cp_e_label)

                        ####### Build labels for the subquery entrance
                        if len(segment_switch_labels) == 0 and in_subquery:
                            # the first vertex of the subquery that has been traveled,
                            # denoting need to going to the subquery
                            segment_switch_labels.append(1)
                        else:
                            segment_switch_labels.append(0)

                        ####### Build object labels
                        aqg_obj_labels.append(next_v_class)
                        aqg_obj_labels.append(v_ids[current_vertex])
                        aqg_obj_labels.append(next_e_class)

                        ####### Build object labels for vertex instances
                        if next_v_class == V_CLASS_IDS["var"]:
                            next_v_instance_label = -1
                        else:
                            next_v_name = aqg.get_vertex_instance(next_vertex)[-1]
                            next_v_instance_label = get_instance_id_label(next_v_name,
                                                                          instance_pool_with_gold["vertex"][next_v_class])
                        v_instance_obj_labels.append(next_v_instance_label)  # index

                        ####### Build object labels for vertex instances
                        next_e_name = aqg.get_edge_instance(edge)[-1]
                        next_e_name = get_origin_name(next_e_name)
                        next_e_class = next_e_class - 1 if next_e_class % 2 == 1 else next_e_class
                        next_e_instance_label = get_instance_id_label(next_e_name,
                                                                      instance_pool_with_gold["edge"][next_e_class])
                        e_instance_obj_labels.append(next_e_instance_label)  # index

                        ####### Record index of vertex and edges
                        v_ids[next_vertex] = len(v_ids)
                        if edge not in e_ids:
                            e_ids[edge] = len(e_ids)                # add the inverse edge, because it is possible
                            e_ids[get_inv_edge(edge)] = len(e_ids)  # that the two edges are the same but in different directions

                    dfs(next_vertex, layer)

        aqg_obj_labels = []
        v_instance_obj_labels = []
        e_instance_obj_labels = []

        v_copy_labels = []
        e_copy_labels = []
        segment_switch_labels = []

        in_subquery = True      # Used to record whether it is currently in a subquery
        visit = set()
        visit.add(v_start)

        if not v_ids and not e_ids:
            # This is the main query
            in_subquery = False
            v_ids = dict()      # Re-record the index of vertices that have been traversed
            e_ids = dict()      # Re-record the index of edges that have been traversed, including inverse edges
            v_ids[v_start] = 0

            v_start_label = aqg.get_vertex_label(v_start)

            aqg_obj_labels.append(v_start_label)
            v_instance_obj_labels.append(-1)

            v_copy_labels.append(-1)
            segment_switch_labels.append(0)

        dfs(v_start, layer)
        return aqg_obj_labels, v_instance_obj_labels, e_instance_obj_labels, \
               v_copy_labels, e_copy_labels, segment_switch_labels, v_ids, e_ids

    def check_labels(self, labels):
        results = [["av", self.v_classes[labels[0]]]]
        for i, label in enumerate(labels[1:]):
            if i % 3 == 0:
                results.append(["av", self.v_classes[label]])
            elif i % 3 == 1:
                results.append(["sv", label])
            else:
                results.append(["ae", self.e_classes[label]])
        return results
