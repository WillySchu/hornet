"""parser/stream.py's TokenStream: a file's tokens and where the parser is in them, which is all the
state parsing has."""

import pytest

from lexer import Lexer, TokenType
from parser.errors import ParseError
from parser.stream import TokenStream


def _stream(source: str) -> TokenStream:
    return TokenStream(Lexer(source, 'f.ht').tokenize())


def test_it_needs_tokens_that_end():
    with pytest.raises(ValueError, match="tokens must have non zero length"):
        TokenStream([])
    with pytest.raises(ValueError, match="tokens must be terminated by an EOF"):
        TokenStream(_stream("a").tokens[:-1])


def test_looking_and_moving():
    stream = _stream("a + 1")
    assert stream.previous() is None and stream.current().val == 'a' and stream.peek(1).type == TokenType.PLUS
    assert stream.check(TokenType.IDENTIFIER) and not stream.check(TokenType.NUMBER)
    assert stream.check(TokenType.NUMBER, TokenType.IDENTIFIER)             # any of several
    assert stream.advance().val == 'a' and stream.previous().val == 'a'
    assert not stream.match(TokenType.NUMBER) and stream.pos == 1            # no match: nothing moves
    assert stream.match(TokenType.PLUS) and stream.expect(TokenType.NUMBER).val == '1'


def test_the_end_is_never_passed():
    stream = _stream("a")
    while not stream.at_end():
        stream.advance()
    end = stream.pos
    assert stream.advance().type == TokenType.EOF and stream.pos == end      # advancing stays put
    assert stream.peek(100).type == TokenType.EOF                            # looking past it sees it
    assert not stream.check(TokenType.EOF)                                   # ... and nothing is "there"


def test_expecting_what_is_not_there():
    stream = _stream("a b")
    stream.advance()
    with pytest.raises(ParseError, match="Expected number, got identifier 'b'") as e:
        stream.expect(TokenType.NUMBER)
    assert (e.value.line, e.value.col) == (1, 3) and stream.pos == 1
    with pytest.raises(ParseError, match="^a name for it"):
        stream.expect(TokenType.NUMBER, "a name for it")


def test_trying_and_taking_back():
    stream = _stream("*P q")
    stream.could_be_multiplication.add(0)
    mark = stream.mark()
    stream.advance()
    stream.advance()
    stream.could_be_multiplication.add(stream.pos)       # noted by position while reading ahead
    stream.reset(mark)
    assert stream.pos == 0 and stream.current().type == TokenType.STAR
    assert stream.could_be_multiplication == {0, 2}      # ... and still true of those tokens


def test_a_statement_ends_its_line():
    stream = _stream("a\nb c\n")
    stream.advance()
    stream.expect_statement_end()                         # a newline follows
    stream.skip_newlines()
    stream.advance()
    with pytest.raises(ParseError, match="Expected the end of the line after this statement, got identifier 'c'"):
        stream.expect_statement_end()
    stream.advance()
    stream.advance()
    stream.expect_statement_end()                         # a newline was just passed
    assert _stream("").expect_statement_end() is None     # the end of input ends one too


def test_a_literal_s_text_and_where_a_bad_escape_is():
    stream = _stream("'a\\tb' 'bad \\q'")
    assert stream.literal_text(stream.advance()) == 'a\tb'
    with pytest.raises(ParseError, match="Unknown escape") as e:
        stream.literal_text(stream.current())
    assert (e.value.line, e.value.col) == (1, 13)         # the backslash, not the literal's start
