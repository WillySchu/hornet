" Vim syntax file
" Language: Hornet
" Filenames: *.ht
" Last Change: 2026-09-29

if exists("b:current_syntax")
  finish
endif

syn case match

" Literals
syn match   hornetNumber  /\<\d\+\%(\.\d\+\)\?\>/
syn keyword hornetBoolean true false
syn keyword hornetConstant none

syn match   hornetEscape /\\x\x\x\|\\./ contained
syn region  hornetString start=/'/ skip=/\\./ end=/'/ contains=hornetEscape
" Byte literals: one byte in double quotes.
syn region  hornetByte   start=/"/ skip=/\\./ end=/"/ contains=hornetEscape

" Types. byte is uint8 and int64 is int.
syn keyword hornetType int int8 uint8 int32 int64 byte bool str dict

" Declarations
syn keyword hornetKeyword def return
syn keyword hornetStructure type struct
syn keyword hornetInclude import from as
syn keyword hornetStorageClass extern intrinsic const

" Control flow
syn keyword hornetConditional if elif else match is
syn keyword hornetRepeat for while in
syn keyword hornetStatement break continue

" Operators. When matches start at the same column the later one wins, so
" the guards keep '=' out of '==' and '<<' out of '<<='.
syn match   hornetOperator /[+\-*\/%~&|^<>]/
syn match   hornetAssignment /<<=\|>>=\|+=\|-=\|\*=\|\/=\|%=\|&=\||=\|\^=\|[=!<>]\@<!==\@!/
syn match   hornetOperator /==\|!=\|<=\|>=\|<<=\@!\|>>=\@!/
syn keyword hornetLogicalOperator and or not

" Calls. Builtins have their own group.
syn match   hornetFunction /\h\w*\ze\s*(/
syn keyword hornetBuiltin print len append del bytes

" Declared names (after the operator and call matches, so they win).
syn match   hornetTypeName /\%(\<type\s\+\)\@<=\h\w*/
syn match   hornetFunctionDef /\%(\<\%(def\|extern\|intrinsic\)\>[^(#]*\)\@<=\<\h\w*\ze\s*(/
" Method receiver, including the '*' of a pointer receiver: def m(*self, ...).
syn match   hornetReceiver /\%(\<def\>[^(#]*(\s*\)\@<=\*\?\h\w*\ze\s*[,)]/

syn match   hornetDelimiter /[(),:;\[\]{}.]/

syn match   hornetComment "#.*$" contains=@Spell

hi def link hornetNumber Number
hi def link hornetBoolean Boolean
hi def link hornetConstant Constant
hi def link hornetString String
hi def link hornetByte Character
hi def link hornetEscape SpecialChar

hi def link hornetType Type
hi def link hornetTypeName Typedef
hi def link hornetStructure Structure
hi def link hornetInclude Include
hi def link hornetStorageClass StorageClass

hi def link hornetKeyword Keyword
hi def link hornetConditional Conditional
hi def link hornetRepeat Repeat
hi def link hornetStatement Statement

hi def link hornetAssignment Operator
hi def link hornetOperator Operator
hi def link hornetLogicalOperator Operator

hi def link hornetFunctionDef Function
hi def link hornetFunction Function
hi def link hornetBuiltin Function
hi def link hornetReceiver Special
hi def link hornetDelimiter Delimiter

hi def link hornetComment Comment

let b:current_syntax = "hornet"
