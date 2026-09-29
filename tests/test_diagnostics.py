"""Tests for diagnostics.py and error reporting across the pipeline."""

import subprocess
import sys
from pathlib import Path

import pytest

import semantic
from compile import compile_to_asm
from diagnostics import CompileError, InternalCompilerError, format_error, format_errors, run_cli
from ir.errors import IRError
from lexer import LexError, lex
from modules import ModuleError
from parser import ParseError, Parser

ROOT = Path(__file__).resolve().parent.parent


def _write(tmp_path: Path, name: str, source: str) -> Path:
    path = tmp_path / name
    path.write_text(source)
    return path


def test_format_error_shows_location_source_and_caret(tmp_path):
    path = _write(tmp_path, 'p.ht', 'def int main():\n    int x = $\n')
    with pytest.raises(LexError) as info:
        lex(str(path))
    lines = format_error(info.value).split('\n')
    assert lines[0].endswith("p.ht:2:13: error: Unexpected character '$'")
    assert lines[1] == '        int x = $'
    assert lines[2] == '    ' + ' ' * 12 + '^'


def test_str_keeps_legacy_position_suffix():
    err = CompileError('bad thing', 'f.ht', 3, 7)
    assert str(err) == 'bad thing at line 3, column 7'
    assert err.message == 'bad thing'


def test_lexer_raises_compile_error(tmp_path):
    path = _write(tmp_path, 'p.ht', 'def int main():\n    return 0 @\n')
    with pytest.raises(CompileError):
        lex(str(path))


def test_parse_error_describes_tokens_and_records_file(tmp_path):
    path = _write(tmp_path, 'p.ht', 'def int main(:\n    return 0\n')
    with pytest.raises(ParseError) as info:
        Parser(lex(str(path))).parse_program()
    assert "got ':'" in info.value.message
    assert info.value.file == str(path)
    assert (info.value.line, info.value.col) == (1, 14)


def test_semantic_error_in_imported_module_names_that_file(tmp_path):
    _write(tmp_path, 'util.ht', "def int f():\n    return 'x'\n")
    entry = _write(tmp_path, 'main.ht', "import 'util'\ndef int main():\n    return util.f()\n")
    with pytest.raises(semantic.SemanticError) as info:
        compile_to_asm(str(entry))
    assert Path(info.value.file).name == 'util.ht'
    assert info.value.line == 2


def test_module_error_has_file_and_clean_message(tmp_path):
    entry = _write(tmp_path, 'main.ht', "import 'nope'\ndef int main():\n    return 0\n")
    with pytest.raises(ModuleError) as info:
        compile_to_asm(str(entry))
    assert info.value.file == str(entry)
    assert info.value.line == 1
    assert 'at line' not in info.value.message
    assert 'at line 1' in str(info.value)


_THREE_BAD_FUNCTIONS = (
    'def int f():\n'
    '    return true\n'
    '\n'
    'def g():\n'
    '    int y = z\n'
    '\n'
    'def int main():\n'
    "    return 'a'\n"
)


def test_errors_collected_per_function(tmp_path):
    entry = _write(tmp_path, 'p.ht', _THREE_BAD_FUNCTIONS)
    with pytest.raises(semantic.SemanticError) as info:
        compile_to_asm(str(entry))
    errors = info.value.errors
    assert [e.line for e in errors] == [2, 5, 8]
    assert str(info.value) == str(errors[0])
    assert format_errors(info.value).count('error:') == 3


def test_single_error_is_not_wrapped(tmp_path):
    entry = _write(tmp_path, 'p.ht', 'def int main():\n    return true\n')
    with pytest.raises(semantic.SemanticError) as info:
        compile_to_asm(str(entry))
    assert type(info.value) is semantic.SemanticError


def test_error_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(semantic, 'MAX_ERRORS', 2)
    entry = _write(tmp_path, 'p.ht', _THREE_BAD_FUNCTIONS)
    with pytest.raises(semantic.SemanticError) as info:
        compile_to_asm(str(entry))
    assert len(info.value.errors) == 2


def test_internal_errors_are_not_compile_errors():
    assert issubclass(IRError, InternalCompilerError)
    assert not issubclass(IRError, CompileError)


def test_run_cli_user_error_exits_1(capsys):
    def action():
        raise CompileError('nope', None, 1, 1)
    with pytest.raises(SystemExit) as info:
        run_cli(action)
    assert info.value.code == 1
    assert '<input>:1:1: error: nope' in capsys.readouterr().err


def test_run_cli_internal_error_exits_2(capsys):
    def action():
        raise IRError('broken')
    with pytest.raises(SystemExit) as info:
        run_cli(action)
    assert info.value.code == 2
    err = capsys.readouterr().err
    assert err.startswith('internal compiler error: IRError: broken')
    assert 'Traceback' in err


def test_run_cli_traceback_flag_reraises():
    def action():
        raise CompileError('nope')
    with pytest.raises(CompileError):
        run_cli(action, show_traceback=True)


def test_build_cli_reports_without_traceback(tmp_path):
    entry = _write(tmp_path, 'p.ht', _THREE_BAD_FUNCTIONS)
    result = subprocess.run(
        [sys.executable, str(ROOT / 'build.py'), str(entry), '-o', str(tmp_path / 'out')],
        capture_output=True, text=True, cwd=tmp_path,
    )
    assert result.returncode == 1
    assert 'Traceback' not in result.stderr
    assert result.stderr.startswith('p.ht:2:5: error:')
    assert result.stderr.count('error:') == 3
