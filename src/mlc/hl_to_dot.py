from mlc import hlir

class HLToDotVisitor:

    def __init__(self):
        self.dot = """\
digraph G {
  newrank = true;
  rankdir = LR;
"""

    def visit(self, x):
        supported_nodes = ["Array", "Operation"]
        node_name = type(x).__name__
        if node_name in supported_nodes:
            eval("self.visit_%s(x)" % node_name)
        else:
            raise Exception("Unsupported HL IR node: %s" % node_name)

    def visit_Array(self, x: hlir.Array):
        self.dot += """  "%s" [ style = filled; fillcolor = pink; shape = record; label="%s"; ]\n""" % (hex(id(x)), "%s (%s) | Array %r" % (x.name, x.type.name, list(x.shape)))

    def visit_Operation(self, x: hlir.Operation):
        for n, arg in enumerate(x.args):
            self.visit(arg)
            self.dot += """  "%s" -> "%s" [ arrowhead = vee; style = solid; label = "%s"; ]\n""" % (hex(id(arg)), hex(id(x)), "arg %d" % n)
        # TODO: add f32 as a type to Operation
        self.dot += """  "%s" [ style = filled; fillcolor = white; shape = record; label="%s"; ]\n""" % (hex(id(x)), "%s (%s) | Op %r" \
                    % (x.op_type.name, "f32", list(x.shape)))



def hl_to_dot(ir: hlir.Operation):
    v = HLToDotVisitor()
    v.visit_Operation(ir)
    return v.dot + "}"
