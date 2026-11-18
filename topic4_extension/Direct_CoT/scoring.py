"""Strict structured answer extraction and bounded Python expression evaluator."""
import ast, json, math, operator, re

FUNCS={k:getattr(math,k) for k in ('sqrt','log','log10','exp','sin','cos','tan','asin','acos','atan')}
FUNCS.update(abs=abs)
OPS={ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul,ast.Div:operator.truediv,ast.Pow:operator.pow}

def calculate(expression):
    """No exec/eval, imports, attribute lookup, loops, files, subprocesses or network."""
    if not isinstance(expression,str) or len(expression)>500:raise ValueError('Expression missing or too long')
    tree=ast.parse(expression,mode='eval')
    if len(list(ast.walk(tree)))>100:raise ValueError('Expression too complex')
    def walk(n):
        if isinstance(n,ast.Expression):return walk(n.body)
        if isinstance(n,ast.Constant) and type(n.value) in (float,int):v=float(n.value)
        elif isinstance(n,ast.Name) and n.id in ('pi','e'):v=getattr(math,n.id)
        elif isinstance(n,ast.UnaryOp) and isinstance(n.op,(ast.UAdd,ast.USub)):v=walk(n.operand)*(1 if isinstance(n.op,ast.UAdd) else -1)
        elif isinstance(n,ast.BinOp) and type(n.op) in OPS:
            a,b=walk(n.left),walk(n.right)
            if isinstance(n.op,ast.Pow) and abs(b)>100:raise ValueError('Exponent exceeds bound')
            v=OPS[type(n.op)](a,b)
        elif isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in FUNCS and len(n.args)<=2 and not n.keywords:
            v=FUNCS[n.func.id](*(walk(a) for a in n.args))
        else:raise ValueError('Unsupported Python expression')
        if not isinstance(v,(float,int)) or not math.isfinite(v) or abs(v)>1e100:raise ValueError('Nonfinite or excessive result')
        return v
    return walk(tree)

def parse_answer(text):
    decoder=json.JSONDecoder(); found=[]
    for match in re.finditer(r'\{',text):
        try:
            obj,_=decoder.raw_decode(text[match.start():])
            if isinstance(obj,dict) and 'answer' in obj:found.append(obj)
        except (ValueError,TypeError):pass
    if len(found)!=1:return None,'Expected exactly one answer JSON object'
    obj=found[0];v=obj['answer']
    if isinstance(v,bool):return None,'Boolean is not numeric'
    try:v=float(v)
    except (ValueError,TypeError):return None,'Answer is not numeric'
    if not math.isfinite(v):return None,'Answer not finite'
    if not isinstance(obj.get('unit'),str):return None,'Unit missing'
    obj['answer']=v
    return obj,None

def score(text,q,rtol=None):
    obj,error=parse_answer(text)
    if error:return {'correct':False,'numeric_correct':False,'format_valid':False,'unit_correct':False,'error':error}
    # Exact standardized target-unit string avoids ambiguous conversions.
    unit_ok=obj['unit'].strip()==q['target_unit']
    numeric=abs(obj['answer']-q['gold'])<=max(q['atol'],(q['rtol'] if rtol is None else rtol)*abs(q['gold']))
    return {'correct':bool(numeric and unit_ok),'numeric_correct':numeric,'format_valid':True,
            'unit_correct':unit_ok,'prediction':obj['answer'],'predicted_unit':obj['unit'],'error':None}
