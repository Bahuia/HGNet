from rules.grammar import AbstractQueryGraph


class Beam():

    def __init__(self, sid: int):
        aqg = AbstractQueryGraph()
        aqg.init_state()

        self.sid = sid
        self.sid_v = -1
        self.sid_e = -1

        self.t = -1
        self.prev_beam_id = -1
        self.prev_v_beam_id = -1
        self.prev_e_beam_id = -1

        self.pred_aqgs = [aqg]

        self.pred_aqg_objs = []

        self.pred_v_copy_objs = []
        self.pred_e_copy_objs = []

        self.pred_v_ins_objs = []
        self.pred_e_ins_objs = []

    @property
    def cur_aqg(self):
        return self.pred_aqgs[-1]

    def update_step(self, t):
        self.t = t

    def update_previous_beam_id(self, b_id):
        self.prev_beam_id = b_id

    def update_vertex_previous_beam_id(self, b_id):
        self.prev_v_beam_id = b_id

    def update_edge_previous_beam_id(self, b_id):
        self.prev_e_beam_id = b_id

    def add_aqg(self, aqg):
        self.pred_aqgs.append(aqg)

    def add_object(self, obj):
        self.pred_aqg_objs.append(obj)

    def add_vertex_instance_object(self, obj):
        self.pred_v_ins_objs.append(obj)

    def add_edge_instance_object(self, obj):
        self.pred_e_ins_objs.append(obj)

    def add_vertex_copy_object(self, obj):
        self.pred_v_copy_objs.append(obj)

    def add_edge_copy_object(self, obj):
        self.pred_e_copy_objs.append(obj)


