Hornet Lang?

as -o test.o test.s
gcc -o test test.o
./test; echo $?


TODO:

Binary Ops:
- Consider replacing ^ with ~
- Exponentiation (either ** or ^)

Compound Assignment Ops:
- +=
- -=
- /=
- \*=
- %=
- <<=
- >>=
- &=
- |=
- ^=

Updates:
- All code paths must return / default return values.
- Ternary.
- For loops.
- Free memory `malloc`ed by string concatenation.
