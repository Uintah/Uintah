#ifndef __UTIL_H__
#define __UTIL_H__

#include <vector>

#include <stdlib.h>
#include <stdarg.h>

#include "mydriver.h"

using std::vector;

/* Converting to/from our C++ data structures to Hypre-required
   structures */
void copyIndex(const Index& subFrom,
               Index* subTo,
               const Counter numDims);
void ToIndex(const vector<Counter>& from,
             Index* to,
             const Counter numDims);
void ToIndex(const vector<int>& from,
             Index* to,
             const Counter numDims);

/* Printouts */
void Print(char *fmt, ...);
void Proc0Print(char *fmt, ...);
template<class T>
void printIndex(const vector<T>& a);

/* Faces, boxes, extents */
void faceExtents(const vector<int>& ilower,
                 const vector<int>& iupper,
                 const Counter dim,
                 const int side,
                 vector<int>& faceLower,
                 vector<int>& faceUpper);
Counter numCellsInBox(const vector<int>& x,
                  const vector<int>& y);
void IndexPlusPlus(const vector<int>& ilower,
                   const vector<int>& iupper,
                   const vector<bool>& active,
                   vector<int>& sub,
                   bool& eof);

/* I/O, MPI */
int clean(void);
void serializeProcsBegin(void);
void serializeProcsEnd(void);

/* vector<int>, vector<double> useful operations */
template<class T, class S>
void pointwiseAdd(const vector<T>& x,
                  const vector<S>& y,
                  vector<S>& result);
template<class T, class S>
void pointwiseMult(const vector<T>& i,
                   const vector<S>& h,
                   vector<S>& result);
template<class T>
void scalarMult(const vector<T>& x,
                const T& h,
                vector<T>& result);
template<class T, class S>
void pointwiseDivide(const vector<S>& x,
                     const vector<T>& y,
                     vector<S>& result);
template<class T> T prod(const vector<T>& x);

/* Miscellaneous utilities */
double roundDigit(const double& x,
                  const Counter d /* = 0 */);
IntMatrix grayCode(const Counter n,
                   const vector<Counter>& k);

#endif // __UTIL_H__
