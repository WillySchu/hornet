"""Composite type x context x value-shape programs, for tests/test_shape_matrix.py."""

DECLS = """type C struct:
    int r
type D struct:
    int k
type U is C | D
type V is dict[int]int | []int
type N is str | none
type P struct:
    int x
    []int s
type H struct:
    [2]int fa
    []int fs
    P fp
    dict[int]int fd
    str ft
    U fu
def [2]int mk_a():
    return [1, 2]
def []int mk_s():
    return []int[1, 2]
def P mk_p():
    return P(1, []int[2])
def U mk_u():
    return C(1)
def dict[int]int mk_d():
    return dict[int]int{1: 2}
def str mk_t():
    return 'ab'
def V mk_v():
    return dict[int]int{1: 2}
def N mk_n():
    return none
"""

# key: (type text, literal, holder field, printed canonical value, comparable)
TYPES = {
    'a': ('[2]int', '[1, 2]', 'fa', '[2]int[1, 2]', True),
    's': ('[]int', '[]int[1, 2]', 'fs', '[]int[1, 2]', False),
    'p': ('P', 'P(1, []int[2])', 'fp', 'P(x: 1, s: []int[2])', False),
    'u': ('U', 'C(1)', 'fu', 'C(r: 1)', False),
    'd': ('dict[int]int', 'dict[int]int{1: 2}', 'fd', 'dict[int]int{1: 2}', False),
    't': ('str', "'ab'", 'ft', 'ab', True),
    'v': ('V', 'dict[int]int{1: 2}', None, 'dict[int]int{1: 2}', False),  # a sum with a dict variant
    'n': ('N', 'none', None, 'none', False),  # a sum holding its `none` variant
}


def _holder(k: str) -> str:
    vals = {'a': '[1, 2]', 's': '[]int[1, 2]', 'p': 'P(1, []int[2])', 'd': 'dict[int]int{1: 2}', 't': "'ab'", 'u': 'C(1)'}
    return "H(" + ", ".join(vals[x] for x in 'aspdtu') + ")"


def shapes(k: str):
    """(name, setup lines, expression, printed value) for each way to produce a value of type k."""
    ty, lit, field, canon, _ = TYPES[k]
    out = [
        ('var', [f"{ty} src = {lit}"], 'src', canon),
        ('field', [f"H h = {_holder(k)}"], f"h.{field}", canon) if field else None,
        ('index', [f"[1]{ty} arr = [{lit}]"], 'arr[0]', canon),
        ('deref', [f"{ty} src = {lit}", f"*{ty} ptr = &src"], '*ptr', canon),
        ('literal', [], lit, canon),
        ('call', [], f"mk_{k}()", canon),
    ]
    out = [x for x in out if x is not None]
    if k == 's':
        out.append(('slice', ["[3]int a3 = [1, 2, 9]"], 'a3[0:2]', canon))
        out.append(('empty', [], '[]', '[]int[]'))
    if k == 't':
        out.append(('slice', ["str t3 = 'abX'"], 't3[0:2]', canon))
        out.append(('from_bytes', ["[]byte bsrc = bytes('ab')"], 'str(bsrc)', canon))
        out.append(('bytes_round_trip', [], 'str(bytes(mk_t()))', canon))
    if k == 'u':
        out.append(('variant_var', ["C cv = C(1)"], 'cv', canon))
    if k == 'v':
        out.append(('variant_var', ["dict[int]int dv = dict[int]int{1: 2}"], 'dv', canon))
    return out


def contexts(k: str, expr: str):
    """(name, setup lines, statements, expected lines or None when not checked)."""
    ty, lit, field, canon, comparable = TYPES[k]
    out = [
        ('vardecl', [], [f"{ty} dst = {expr}", "print(dst)"], ['{v}']),
        ('assign', [f"{ty} dst = {lit}"], [f"dst = {expr}", "print(dst)"], ['{v}']),
        ('field_assign', [f"H h2 = {_holder(k)}"], [f"h2.{field} = {expr}", f"print(h2.{field})"], ['{v}']) if field else None,
        ('array_index_assign', [f"[1]{ty} a2 = [{lit}]"], [f"a2[0] = {expr}", "print(a2[0])"], ['{v}']),
        ('slice_index_assign', [f"[]{ty} s2 = []{ty}[{lit}]"], [f"s2[0] = {expr}", "print(s2[0])"], ['{v}']),
        ('deref_assign', [f"{ty} dst = {lit}", f"*{ty} pd = &dst"], [f"*pd = {expr}", "print(dst)"], ['{v}']),
        ('call_arg', [], [f"sink({expr})"], ['{v}']),
        ('print', [], [f"print({expr})"], ['{v}']),
        ('append', [f"[]{ty} ap"], [f"ap = append(ap, {expr})", "print(ap[0])"], ['{v}']),
        ('dict_value', [f"dict[int]{ty} m = dict[int]{ty}{{}}"], [f"m[0] = {expr}", "print(m[0])"], ['{v}']),
        ('return', [], ["print(ret())"], ['{v}']),
        ('statement', [], [expr, "print(1)"], ['1']),
    ]
    out = [x for x in out if x is not None]
    if k in 'asd':
        out.append(('for_in', [], [f"for x in {expr}:", "    print(x)"], None))
    if comparable:
        out.append(('equality', [], [f"if {expr} == {expr}:", "    print(1)"], ['1']))
    return out


def programs():
    """Yield (name, source, expected stdout or None)."""
    for k, (ty, lit, field, canon, _) in TYPES.items():
        for shape, shape_setup, expr, value in shapes(k):
            for ctx, ctx_setup, stmts, expected in contexts(k, expr):
                funcs = [f"def sink({ty} x):", "    print(x)"]
                if ctx == 'return':
                    funcs += [f"def {ty} ret():"] + [f"    {line}" for line in shape_setup] + [f"    return {expr}"]
                    body_setup = []
                else:
                    body_setup = shape_setup
                lines = funcs + ["def int main():"]
                lines += [f"    {line}" for line in body_setup + ctx_setup + stmts]
                lines.append("    return 0")
                exp = None if expected is None else '\n'.join(e.format(v=value) for e in expected) + '\n'
                yield f"{k}-{shape}-{ctx}", DECLS + '\n'.join(lines) + '\n', exp
