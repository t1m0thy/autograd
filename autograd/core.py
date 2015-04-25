from __future__ import absolute_import
import warnings
import inspect
import operator as op
import types
import math
import numpy as np

def grad(fun, argnum=None, argname=None):
    if (argnum is not None) and (argname is not None):
        raise ValueError("Specify an argnum OR an argname (not both).")

    def gradfun(*args,**kwargs):
        grad_dict = backward_pass(*forward_pass(fun,args,kwargs,argnum,argname))
        return unpack_result(fun,grad_dict,argnum,argname)

    return gradfun

def forward_pass(fun, args, kwargs, argnum=None, argname=None):
    if argnum is not None:
        arglist = getargspec(fun).args
        argname = getargspec(fun).args[argnum] if len(arglist) > argnum else None
    kwargs = getcallargs(fun,*args,**kwargs)
    args_to_replace = kwargs.keys() if argname is None else {argname}

    tape = CalculationTape()
    start_nodes, new_kwargs = replace_args_with_nodes(kwargs, args_to_replace, tape)
    argidx = lambda (argname, argval): getargspec(fun).args.index(argname)
    new_args = [v for _, v in sorted(new_kwargs.items(),key=argidx)]
    end_node = fun(*new_args)
    return tape, start_nodes, end_node

def backward_pass(tape, start_nodes, end_node):
    if not isinstance(end_node, Node) or tape not in end_node.tapes:
        warnings.warn("Output seems independent of input. Returning zero gradient.")
        return {argname:zeros_like(start_node)
                for argname, start_node in start_nodes.iteritems()}
    if not type(end_node) is FloatNode:
        try:
            end_node = FloatNode.cast(end_node)
        except TypeError:
            raise TypeError("Output type {0} can't be cast to float. ".format(type(end_node.value))
                            + "Function grad requires a scalar-valued function. "
                              "Try jacobian or elementwise_grad.")

    start_names = {start_node.tapes[tape]:argname
                   for argname, start_node in start_nodes.iteritems()}
    end_node.tapes[tape].outgrads = [1.0]
    tape.complete = True
    op_list = list(tape)
    grad_dict = {}
    while op_list:
        node = op_list.pop()
        if node.outgrads:
            cur_outgrad = node.sum_outgrads()
            assert type(new_node(getval(cur_outgrad))) == node.node_type, \
                "Types are {0} and {1}".format(type(new_node(getval(cur_outgrad))), node.node_type)
            for gradfun, parent in node.parent_grad_ops:
                og = cast_to_node_type(gradfun(cur_outgrad), parent.node_type)
                parent.outgrads.append(og)
            if node in start_names:
                grad_dict[start_names[node]] = cur_outgrad
    return grad_dict

def replace_args_with_nodes(arg_bindings, args_to_replace, tape):
    def makenode(argval):
        return new_node(safe_type(getval(argval)), [tape])
    start_nodes = {k:makenode(arg_bindings[k]) for k in args_to_replace}
    new_args = {k:merge_tapes(start_nodes[k], v) if k in args_to_replace else v
                for k, v in arg_bindings.iteritems()}
    return start_nodes, new_args

def unpack_result(fun, grad_dict, argnum, argname):
    if argnum is not None:
        argname = getargspec(fun).args[argnum]
    if argname is not None:
        return grad_dict[argname]
    return grad_dict

def getargspec(fun):
    if isinstance(fun,primitive):
        fun = fun.fun
    return inspect.getargspec(fun)

def getcallargs(fun,*args,**kwargs):
    if isinstance(fun,primitive):
        fun = fun.fun
    return inspect.getcallargs(fun,*args,**kwargs)

def cast_to_node_type(x, node_type):
    if type(new_node(getval(x))) is not node_type:
        return node_type.cast(x)
    else:
        return x

class primitive(object):
    """
    Wraps a function so that its gradient can be specified and its invocation
    can be recorded. For examples, see the docs."""
    def __init__(self, fun):
        self.fun = fun
        self.grads = {}
        self.zero_grads = set()
        self.__name__ = fun.__name__
        self.__doc__ = fun.__doc__

    def gradmaker(self, argnum, *args, **kwargs):
        try:
            return self.grads[argnum](*args, **kwargs)
        except KeyError:
            if self.grads == {}:
                raise NotImplementedError("Gradient of {0} not yet implemented."
                                          .format(self.fun, argnum))
            raise NotImplementedError("Gradient of {0} w.r.t. arg number {1} not yet implemented."
                                      .format(self.fun, argnum))

    def defgrad(self, gradmaker, argnum=0):
        self.grads[argnum] = gradmaker

    def defgrad_is_zero(self, argnums=(0,)):
        for argnum in argnums:
            self.zero_grads.add(argnum)

    def __call__(self, *args, **kwargs):
        argvals = list(args)
        ops = []
        tapes = set()
        for i, arg in enumerate(args):
            if isinstance(arg, Node):
                argvals[i] = arg.value
                if i in self.zero_grads: continue
                for tape, parent_rnode in arg.tapes.iteritems():
                    if not tape.complete:
                        ops.append((tape, i, parent_rnode))
                        tapes.add(tape)

        result = self.fun(*argvals, **kwargs)
        if result is NotImplemented: return result
        if ops:
            result = new_node(result, tapes)
            for tape, argnum, parent in ops:
                gradfun = self.gradmaker(argnum, result, *args, **kwargs)
                rnode = result.tapes[tape]
                rnode.parent_grad_ops.append((gradfun, parent))
        return result

    def __get__(self, obj, objtype):
        return types.MethodType(self, obj, objtype)

@primitive
def merge_tapes(x, y): return x
merge_tapes.defgrad(lambda ans, x, y : lambda g : g)
merge_tapes.defgrad(lambda ans, x, y : lambda g : g, argnum=1)

def new_node(value, tapes=[]):
    try:
        return Node.type_mappings[type(value)](value, tapes)
    except KeyError:
        raise TypeError("Can't differentiate wrt {0}".format(type(value)))

def zeros_like(value):
    if isinstance(value, Node):
        return value.zeros_like(value)
    else:
        return new_node(value, []).zeros_like(value)

class ReverseNode(object):
    __slots__ = ['parent_grad_ops', 'outgrads', 'node_type']
    def __init__(self, node_type):
        self.parent_grad_ops = []
        self.outgrads = []
        self.node_type = node_type

    def sum_outgrads(self):
        return self.node_type.sum_outgrads(self.outgrads)

class Node(object):
    __slots__ = ['value', 'tapes']
    type_mappings = {}
    def __init__(self, value, tapes):
        self.value = value
        self.tapes = {}
        for tape in tapes:
            new_rnode = ReverseNode(type(self))
            tape.append(new_rnode)
            self.tapes[tape] = new_rnode

    @staticmethod
    def sum_outgrads(outgrads):
        return sum(outgrads[1:], outgrads[0])

@primitive
def cast(value, caster):
    return caster(value)
cast.defgrad(lambda *args: I)

getval = lambda x : x.value if isinstance(x, Node) else x

class CalculationTape(list):
    def __init__(self):
        self.complete = False

    def __hash__(self):
        return id(self)

class FloatNode(Node):
    __slots__ = []
    @staticmethod
    def zeros_like(value):
        return 0.0
    @staticmethod
    def cast(value):
        return cast(value, cast_to_float)

Node.type_mappings[float] = FloatNode

def cast_to_float(x):
    if np.iscomplexobj(x):
        x = np.real(x)
    return float(x)

class ComplexNode(FloatNode):
    @staticmethod
    def zeros_like(value):
        return 0.0 + 0.0j
    @staticmethod
    def cast(value):
        return cast(value, cast_to_complex)

def cast_to_complex(value):
    if isinstance(value, np.ndarray):
        return complex(value[()])
    else:
        return complex(value)
Node.type_mappings[complex] = ComplexNode

def safe_type(value):
    if isinstance(value, int):
        warnings.warn("Casting int to float to handle differentiation.")
        return float(value)
    else:
        return value

differentiable_ops = ['__add__', '__sub__', '__mul__', '__pow__', '__div__', '__mod__',
                      '__neg__', '__radd__', '__rsub__', '__rmul__', '__rpow__',
                      '__rdiv__', '__rmod__']
nondifferentiable_ops = ['__eq__', '__ne__', '__gt__', '__ge__', '__lt__', '__le__',]
for float_op in differentiable_ops + nondifferentiable_ops:
    setattr(FloatNode, float_op, primitive(getattr(float, float_op)))

FloatNode.__dict__['__neg__'].defgrad(lambda ans, x : op.neg)

for comp_op in nondifferentiable_ops:
    FloatNode.__dict__[comp_op].defgrad_is_zero(argnums=(0, 1))

# These functions will get clobbered when autograd.numpy is imported.
# They're here to allow the use of autograd without numpy.
I = lambda g: g
FloatNode.__dict__['__add__'].defgrad(lambda ans, x, y : I)
FloatNode.__dict__['__add__'].defgrad(lambda ans, x, y : I, argnum=1)
FloatNode.__dict__['__mul__'].defgrad(lambda ans, x, y : lambda g : y * g)
FloatNode.__dict__['__mul__'].defgrad(lambda ans, x, y : lambda g : x * g, argnum=1)
FloatNode.__dict__['__sub__'].defgrad(lambda ans, x, y : I)
FloatNode.__dict__['__sub__'].defgrad(lambda ans, x, y : op.neg, argnum=1)
FloatNode.__dict__['__div__'].defgrad(lambda ans, x, y : lambda g : g / y)
FloatNode.__dict__['__div__'].defgrad(lambda ans, x, y : lambda g : - g * x / y**2, argnum=1)
FloatNode.__dict__['__pow__'].defgrad(lambda ans, x, y : lambda g : g * y * x ** (y - 1))
FloatNode.__dict__['__pow__'].defgrad(lambda ans, x, y : lambda g : g * log(x) * x ** y, argnum=1)
FloatNode.__dict__['__mod__'].defgrad(lambda ans, x, y : I)
FloatNode.__dict__['__mod__'].defgrad(lambda ans, x, y : lambda g : -g * floor(x/y), argnum=1)

log = primitive(math.log)
log.defgrad(lambda ans, x : lambda g : g / x)
floor = primitive(math.floor)
floor.defgrad_is_zero()

def swap_args(grads):
    grad_0, grad_1 = grads[1], grads[0]
    return {0 : lambda ans, y, x : grad_0(ans, x, y),
            1 : lambda ans, y, x : grad_1(ans, x, y)}

FloatNode.__dict__['__radd__'].grads = swap_args(FloatNode.__dict__['__add__'].grads)
FloatNode.__dict__['__rmul__'].grads = swap_args(FloatNode.__dict__['__mul__'].grads)
FloatNode.__dict__['__rsub__'].grads = swap_args(FloatNode.__dict__['__sub__'].grads)
FloatNode.__dict__['__rdiv__'].grads = swap_args(FloatNode.__dict__['__div__'].grads)
FloatNode.__dict__['__rpow__'].grads = swap_args(FloatNode.__dict__['__pow__'].grads)
FloatNode.__dict__['__rmod__'].grads = swap_args(FloatNode.__dict__['__mod__'].grads)
