"""Methods on an enum, written after its members as a struct's are after its fields: `self` is the
value, or with `*self` a pointer to it, and a call is resolved by the receiver's type."""

import pytest

from diagnostics import CompileError
from dump import dump as dump_stage
from modules import discover_modules
from semantic import SemanticError, analyze
from build import build_executable, executable_name
from tests.targets import on_every_target, run_binary
from tests.test_compiler import GCC_SKIP, assert_program_semantic_error, assert_program_stdout
from typed_ast import dump

LIGHT = (
    "type Light enum:\n"
    "    Red\n"
    "    Amber\n"
    "    Green\n"
    "\n"
    "    def str label(self):\n"
    "        match self:\n"
    "            is Red:\n"
    "                return 'stop'\n"
    "            is Green:\n"
    "                return 'go'\n"
    "            else:\n"
    "                return 'wait'\n"
    "\n"
    "    def Light after(self):\n"                         # returns its own type
    "        return Light((int(self) + 1) % len(Light))\n"
    "\n"
    "    def bool is_before(self, Light other):\n"         # takes another of its type
    "        return int(self) < int(other)\n"
    "\n"
    "    def int steps_to(self, Light target):\n"          # calls other methods, and itself
    "        if self == target:\n"
    "            return 0\n"
    "        return 1 + self.after().steps_to(target)\n"
    "\n"
    "    def advance(*self):\n"                            # changes the variable it is called on
    "        *self = self.after()\n"
    "\n"
    "    def str _secret(self):\n"
    "        return 'private ' + self.label()\n"
    "\n"
    "    def str reveal(self):\n"
    "        return self._secret()\n"
    "\n"
)


@GCC_SKIP
def test_calling_methods():
    assert_program_stdout(
        LIGHT +
        "type Junction struct:\n"
        "    Light north\n"
        "    [2]Light others\n"
        "def Light make():\n"
        "    return Light.Amber\n"
        "def int main():\n"
        "    Light light = Light.Red\n"
        "    print(light.label())\n"
        "    print(Light.Green.label())\n"                 # on a member itself
        "    print(make().label())\n"                      # on a call's result
        "    print(light.after())\n"
        "    print(light.after().after().label())\n"
        "    print(light.is_before(Light.Green))\n"
        "    print(Light.Green.steps_to(Light.Amber))\n"
        "    Junction j = Junction(Light.Green, [Light.Red, Light.Amber])\n"
        "    print(j.north.label() + ' ' + j.others[1].label())\n"   # on a field, and an element
        "    *Light p = &light\n"
        "    print(p.label())\n"                           # through a pointer
        "    light.advance()\n"
        "    print(light)\n"
        "    p.advance()\n"
        "    j.north.advance()\n"
        "    j.others[0].advance()\n"
        "    print(format('{} {} {}', light, j.north, j.others))\n"
        "    print(light.reveal())\n"
        "    print(light._secret())\n"                     # private, but this is its own module
        "    return 0\n",
        "stop\ngo\nwait\nLight.Amber\ngo\ntrue\n2\ngo wait\nstop\nLight.Amber\n"
        "Light.Green Light.Red [2]Light[Light.Amber, Light.Amber]\nprivate go\nprivate go\n",
    )


@pytest.mark.parametrize("source,match", [
    (LIGHT + "def int main():\n    print(Light.Red.missing())\n    return 0\n", "Enum 'Light' has no method 'missing'"),
    (LIGHT + "def int main():\n    print(Light.Red.label(1))\n    return 0\n",
     r"Method 'label' on 'Light' expects 0 argument\(s\), got 1"),
    (LIGHT + "def int main():\n    print(Light.Red.is_before(1))\n    return 0\n",
     "Argument 1 to method 'is_before' on 'Light' should be Light, got int"),
    (LIGHT + "def Light make():\n    return Light.Red\ndef int main():\n    make().advance()\n    return 0\n",
     "Method 'advance' on 'Light' has a pointer receiver, so it needs an addressable receiver"),
    (LIGHT + "def int main():\n    Light.Red.advance()\n    return 0\n",
     "Method 'advance' on 'Light' has a pointer receiver, so it needs an addressable receiver"),
    (LIGHT + "def int main():\n    *Light p = &Light.Red\n    return 0\n",
     "Cannot take the address of the enum member 'Light.Red'"),
    ("def int main():\n    int n = 1\n    print(n.label())\n    return 0\n",
     "Cannot call method 'label' on a value of type int -- methods are only defined on structs and enums"),
    ("type E enum:\n    A\n    B\n    def int A(self):\n        return 1\ndef int main():\n    return 0\n",
     "Method 'A' has the same name as a member of enum 'E'"),
    ("type E enum:\n    A\n    def int f(self):\n        return 1\n    def int f(self):\n        return 2\n"
     "def int main():\n    return 0\n", "Method 'f' is already declared on enum 'E'"),
    ("type E enum:\n    A\n    def int f(self):\n        return missing\ndef int main():\n    return 0\n",
     "undeclared variable 'missing'"),
])
def test_what_is_rejected(source, match):
    assert_program_semantic_error(source, match=match)


def test_members_come_first(tmp_path):
    path = tmp_path / "p.ht"
    path.write_text(
        "type E enum:\n    A\n    def int f(self):\n        return 1\n    B\ndef int main():\n    return 0\n")
    with pytest.raises(CompileError, match="An enum's members come before its methods"):
        dump_stage(str(path), 'tree')


def _module_program(tmp_path, main: str) -> str:
    (tmp_path / "light.ht").write_text(LIGHT)
    (tmp_path / "main.ht").write_text(main)
    return str(tmp_path / "main.ht")


@GCC_SKIP
def test_an_enum_from_another_module_brings_its_methods(tmp_path):
    source = "from 'light' import Light\n\ndef int main():\n    Light l = Light.Red\n    l.advance()\n" \
             "    print(l.label())\n    print(l.reveal())\n    return l.steps_to(Light.Red)\n"
    result = on_every_target(lambda target: _run(tmp_path, source, target))
    assert (result.returncode, result.stdout) == (2, "wait\nprivate wait\n")


def _run(tmp_path, source, target):
    path = _module_program(tmp_path, source)
    exe = tmp_path / executable_name(f"prog-{target}", target)
    build_executable(path, str(exe), target=target)
    return run_binary(target, [exe], capture_output=True, text=True)


def test_a_private_method_stays_in_its_module(tmp_path):
    path = _module_program(tmp_path, "from 'light' import Light\n\ndef int main():\n    print(Light.Red._secret())\n"
                                     "    return 0\n")
    entry, modules = discover_modules(path)
    with pytest.raises(SemanticError, match="Method '_secret' of 'Light' is not visible outside the module that "
                                            "defines the enum"):
        analyze(entry, modules)


def test_a_method_is_a_function_with_the_receiver_first(tmp_path):
    path = tmp_path / "p.ht"
    path.write_text(LIGHT + "def int main():\n    Light l = Light.Red\n    l.advance()\n    print(l.label())\n"
                            "    return 0\n")
    entry, modules = discover_modules(str(path))
    tree = dump(analyze(entry, modules))
    assert "function Light.label(self#" in tree and "function Light.advance(self#" in tree
    assert "Call name=Light.label kind=function : str" in tree
    assert "Function name=label return_type=str" not in dump_stage(str(path), 'tree')  # it is a MethodDef there
    assert "MethodDef receiver_name=self name=advance receiver_is_pointer=true" in dump_stage(str(path), 'tree')
