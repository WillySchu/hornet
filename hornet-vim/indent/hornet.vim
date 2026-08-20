" Vim indent file for Hornet
" Language: Hornet
" Filenames: *.ht

if exists("b:did_indent")
  finish
endif
let b:did_indent = 1

setlocal indentexpr=GetHornetIndent()
setlocal indentkeys=o,O,*<Return>,<>>,<<>,:

if exists("*GetHornetIndent")
  finish
endif

function! GetHornetIndent() abort
  let lnum = v:lnum
  let prev = prevnonblank(lnum - 1)

  if prev <= 0
    return 0
  endif

  let ind = indent(prev)
  let prevline = getline(prev)
  let curline = getline(lnum)

  " Dedent the line following a completed block. Hornet blocks are introduced
  " by a trailing colon and delimited by indentation.
  if curline =~# '^\s*\%(elif\|else\)\>'
    return max([ind - &shiftwidth, 0])
  endif

  " Continue the current block when the previous line opens one.
  if prevline =~# ':\s*$'
    return ind + &shiftwidth
  endif

  return ind
endfunction
