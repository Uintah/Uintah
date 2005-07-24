#ifndef __UTIL_H__
#define __UTIL_H__

#include <vector>

#include <stdlib.h>
#include <stdarg.h>

#include "mydriver.h"

using std::vector;

void copyIndex(const Index& subFrom,
               Index* subTo,
               const int numDims);
void ToIndex(const vector<int>& from,
             Index* to,
             const int numDims);

void Print(char *fmt, ...);
void Proc0Print(char *fmt, ...);

void printIndex(const vector<int>& a);

void printIndex(const vector<double>& x); 

void faceExtents(const vector<int>& ilower,
                 const vector<int>& iupper,
                 const int dim,
                 const int side,
                 vector<int>& faceLower,
                 vector<int>& faceUpper);

void IndexPlusPlus(const vector<int>& ilower,
                   const vector<int>& iupper,
                   const vector<bool>& active,
                   vector<int>& sub,
                   bool& eof);

int clean(void);

void serializeProcsBegin(void);

void serializeProcsEnd(void);

void pointwiseAdd(const vector<int>& x,
                  const vector<int>& y,
                  vector<int>& result);

void pointwiseAdd(const vector<double>& x,
                  const vector<double>& y,
                  vector<double>& result);

void
pointwiseMult(const vector<int>& i,
              const vector<int>& h,
              vector<int>& result);

void scalarMult(const vector<double>& x,
                const double h,
                vector<double>& result);

void pointwiseMult(const vector<int>& i,
                   const vector<double>& h,
                   vector<double>& result);
int prod(const vector<int>& x);

double prod(const vector<double>& x);

#endif // __UTIL_H__
