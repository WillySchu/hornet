"
" Vim syntax file
" Language: Hornet
" Maintainer: generated from the Hornet compiler lexer/parser
" Filenames: *.ht
" Last Change: 2026-08-19

if exists("b:current_syntax")
  finish
endif

syn case match

" ---------------------------------------------------------------------------
" Literals
" ---------------------------------------------------------------------------

syn match   hornetNumber  /\<\d\+\%(\.\d\+\)\?\>/
syn match   hornetBoolean /\<\%(true\|false\)\>/
syn keyword hornetConstant none

syn region  hornetString start=/'/ skip=/\\./ end=/'/ contains=hornetEscape
syn match   hornetEscape /\\./ contained

" ---------------------------------------------------------------------------
" Types
" ---------------------------------------------------------------------------

syn keyword hornetType int bool str
" Array/slice type syntax is structural; the type keywords above still
" highlight the element type inside forms such as [5]int and []int.
syn match   hornetTypeBrackets /[\[\]]/

" ---------------------------------------------------------------------------
" Keywords and control flow
" ---------------------------------------------------------------------------

syn keyword hornetKeyword def return
syn keyword hornetConditional if elif else
syn keyword hornetRepeat while
syn keyword hornetStatement break continue

" ---------------------------------------------------------------------------
 " Operators
" ---------------------------------------------------------------------------

" Compound assignments and plain assignment.
syn match hornetAssignment /<<=\|>>=\|+=\|-=\|\*=\|\/=\|%=\|&=\||=\|\^=\|=/

" Comparisons and shifts.
syn match hornetOperator /==\|!=\|<=\|>=\|<<\|>>/

" Single-character operators. Longer operators above win because they are
" declared first.
syn match hornetOperator /[+\-*\/%~&|^<>]/
syn keyword hornetLogicalOperator and or not

" Builtins and identifiers
" ---------------------------------------------------------------------------

syn keyword hornetBuiltin print len

" A name immediately followed by "(" is a call. This deliberately also
" catches user-defined functions; builtin names have their own stronger
" keyword highlight above.
syn match hornetFunction /\h\w*\ze\s*(/

" ---------------------------------------------------------------------------
" Delimiters
" ---------------------------------------------------------------------------

syn match hornetDelimiter /[(),:;\[\]]/

" ---------------------------------------------------------------------------
" Highlight links
" ---------------------------------------------------------------------------

hi def link hornetNumber Number
hi def link hornetBoolean Boolean
hi def link hornetConstant Constant
hi def link hornetString String
hi def link hornetEscape SpecialChar

hi def link hornetType Type
hi def link hornetTypeBrackets Delimiter

hi def link hornetKeyword Keyword
hi def link hornetConditional Conditional
hi def link hornetRepeat Repeat
hi def link hornetStatement Statement

hi def link hornetAssignment Operator
hi def link hornetOperator Operator
hi def link hornetLogicalOperator Boolean

hi def link hornetBuiltin Function
hi def link hornetFunction Function
hi def link hornetDelimiter Delimiter

let b:current_syntax = "hornet"
