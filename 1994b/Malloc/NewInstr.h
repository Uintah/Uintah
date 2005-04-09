
#ifndef Lib_NewInst_h
#define Lib_NewInst_h 1

#ifdef SCI_LOGNEW
void newinst_push(char*, char*, int);
void newinst_pop(int);

#define NI_PUSH(what) newinst_push(what, __FILE__, __LINE__)
#define NI_POP() newinst_pop(__LINE__)
#else
#define NI_PUSH(what)
#define NI_POP()
#endif

#endif
