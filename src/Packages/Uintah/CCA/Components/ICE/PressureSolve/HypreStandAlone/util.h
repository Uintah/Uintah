#ifndef __UTIL_H__
#define __UTIL_H__

#include "IntMatrix.h"

/* Printouts */
string lineHeader(void);

/* I/O, MPI */
int clean(void);
void serializeProcsBegin(void);
void serializeProcsEnd(void);

/* Miscellaneous utilities */
double roundDigit(const double& x,
                  const Counter d = 0);
IntMatrix grayCode(const Counter n,
                   const Vector<Counter>& k);

#endif // __UTIL_H__
