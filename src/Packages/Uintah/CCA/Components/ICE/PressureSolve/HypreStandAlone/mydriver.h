/*--------------------------------------------------------------------------
 * File: mydriver.h
 *
 * Header file for the test driver for semi-structured matrix interface.
 *
 * Revision history:
 * 19-JUL-2005   Oren   Created to allow rhs(), diffusion() etc. in a separate
 *                      file.
 *--------------------------------------------------------------------------*/

/*================== Library includes ==================*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdarg.h>
#include <vector>
#include "utilities.h"
#include "HYPRE_sstruct_ls.h"
using std::vector;

/*================== Macros ==================*/

#define DEBUG      1        // Debug flag
#define MAX_DIMS   3        // Maximum number of dimensions
#define NUM_VARS   1        // # variable types; we use only cell-centered

/*================== Global variables ==================*/

extern int         numDims;  // Actual number of dimensions
extern int         MYID;     // The same as myid (this proc's id)

/*================== Type definitions ==================*/

typedef int    Index[MAX_DIMS];    // Subscript in d-dimensional space
typedef double Location[MAX_DIMS]; // Data node location in d-dim. space

/*================== Function delcarations ==================*/

/* mydriver.cc */

/* util.cc */
void ToIndex(const vector<int>& from,
             Index* to);

void Print(char *fmt, ...);

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

void pointwiseAdd(const vector<double>& x,
                  const vector<double>& y,
                  vector<double>& result);
void scalarMult(const vector<double>& x,
                const double h,
                vector<double>& result);
void pointwiseMult(const vector<int>& i,
                   const vector<double>& h,
                   vector<double>& result);
int prod(const vector<int>& x);
double prod(const vector<double>& x);
