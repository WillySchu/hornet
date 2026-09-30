" Vim indent file for Hornet
" Language: Hornet
" Filenames: *.ht

if exists("b:did_indent")
  finish
endif
let b:did_indent = 1

setlocal indentexpr=GetHornetIndent()
setlocal indentkeys=o,O,*<Return>,<>>,<<>,:,=elif,=else,=is

if exists("*GetHornetIndent")
  finish
endif

" Indent of the nearest line above `lnum` (indented less than `ind`) matching
" `pattern`, or -1.
function! s:OpenerIndent(lnum, ind, pattern) abort
  let l = a:lnum
  while l > 0
    let i = indent(l)
    if i < a:ind && getline(l) =~# a:pattern
      let text = getline(l)
      " An 'is' arm lines up with the other arms; everything else with its opener.
      return text =~# '^\s*match\>' ? i + &shiftwidth : i
    endif
    if i < a:ind && getline(l) !~# '^\s*$' && getline(l) !~# '^\s*#'
      return -1
    endif
    let l = prevnonblank(l - 1)
  endwhile
  return -1
endfunction

function! GetHornetIndent() abort
  let lnum = v:lnum
  let prev = prevnonblank(lnum - 1)

  if prev <= 0
    return 0
  endif

  let ind = indent(prev)
  let prevline = getline(prev)
  let curline = getline(lnum)

  " Strip a trailing comment before looking at the previous line's shape.
  let prevcode = substitute(prevline, '\s*#.*$', '', '')

  " elif/else, and a match arm after another arm's body, line up with the
  " line that opened the enclosing construct.
  if curline =~# '^\s*\%(elif\|else\)\>'
        \ || (curline =~# '^\s*is\>' && prevcode !~# '^\s*match\>')
    let target = s:OpenerIndent(prev, ind, curline =~# '^\s*is\>' ? '^\s*\%(is\|match\)\>' : '^\s*\%(if\|elif\|else\)\>')
    if target >= 0
      return target
    endif
    return max([ind - &shiftwidth, 0])
  endif

  " Continue the current block when the previous line opens one.
  if prevcode =~# ':\s*$'
    return ind + &shiftwidth
  endif

  " Nothing follows return/break/continue in the same block.
  if prevcode =~# '^\s*\%(return\|break\|continue\)\>'
    return max([ind - &shiftwidth, 0])
  endif

  return ind
endfunction
