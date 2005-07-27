/*--------------------------------------------------------------------------
 * File: util.cc
 *
 * Utility functions: printouts, index conversion, cleanups.
 *
 * Revision history:
 * 20-JUL-2005   Oren   Created.
 *--------------------------------------------------------------------------*/

#include "mydriver.h"
#include "util.h"

#include <utilities.h>
#include <vector>
#include <iostream>

using namespace std;

void 
copyIndex(const Index& subFrom,
          Index* subTo,
          const int numDims)
  /*_____________________________________________________________________
    Function copyIndex:
    Copy an index subFrom to index subTo in numDims-dimensions.
    _____________________________________________________________________*/
{
  for (int d = 0; d < numDims; d++) (*subTo)[d] = subFrom[d];
}

void 
ToIndex(const vector<int>& subFrom,
        Index* subTo,
        const int numDims)
  /*_____________________________________________________________________
    Function ToIndex:
    Convert a vector-type subscript "subFrom" to Index-type "subTo",
    so that we can enjoy the flexibility of vector, but interface to
    Hypre with the correct type (Index).
    _____________________________________________________________________*/
{
  assert(subFrom.size() == numDims);
  for (int d = 0; d < numDims; d++) (*subTo)[d] = subFrom[d];
}

void 
Proc0Print(char *fmt, ...)
{
  if( MYID == 0 ) 
    {
      int vb = 1; /* Verbose level */
      va_list ap;
      va_start(ap, fmt);
      if (vb) {
        //        printf("P%2d: ",MYID);
        fprintf(stderr,"P%2d: ",MYID);
        //        vprintf(fmt, ap);
        vfprintf(stderr, fmt, ap);
      }
      //      fflush(stdout);
      fflush(stderr);
      //      if (vb) {
        //        va_start(ap, fmt);
        //    if (log_file)
        //      vfprintf(log_file, fmt, ap);
        //    if (log_file)
        //      fflush(log_file);
      //      }
      va_end(ap);
    }
}

void 
Print(char *fmt, ...)
  /*_____________________________________________________________________
    Function Print:
    Print an output line on the current processor. Useful to parse MPI
    output.
    _____________________________________________________________________*/
{
  int vb = 1; /* Verbose level */
  va_list ap;
  va_start(ap, fmt);
  if (vb) {
    //    printf("P%2d: ",MYID);
    fprintf(stderr,"P%2d: ",MYID);
    //    vprintf(fmt, ap);
    vfprintf(stderr, fmt, ap);
  }
  //  fflush(stdout);
  fflush(stderr);
  //  if (vb) {
  //    va_start(ap, fmt);
    //    if (log_file)
    //      vfprintf(log_file, fmt, ap);
    //    if (log_file)
    //      fflush(log_file);
  //  }
  va_end(ap);
}

void 
printIndex(const vector<int>& sub) 
{
  /*_____________________________________________________________________
    Function printIndex:
    Print vector-type numDims-dimensional index sub
    _____________________________________________________________________*/
  fprintf(stderr,"[");
  for (int d = 0; d < sub.size(); d++) {
    fprintf(stderr,"%d",sub[d]);
    if (d < sub.size()-1) fprintf(stderr,",");
  }
  fprintf(stderr,"]");
}

void 
printIndex(const vector<double>& x) 
{
  /*_____________________________________________________________________
    Function printIndex:
    Print vector-type numDims-dimensional location x
    _____________________________________________________________________*/
  fprintf(stderr,"[");
  for (int d = 0; d < x.size(); d++) {
    fprintf(stderr,"%+.3lf",x[d]);
    if (d < x.size()-1) fprintf(stderr,",");
  }
  fprintf(stderr,"]");
}

void
faceExtents(const vector<int>& ilower,
            const vector<int>& iupper,
            const int dim,
            const int side,
            vector<int>& faceLower,
            vector<int>& faceUpper)
  /*_____________________________________________________________________
    Function faceExtents:
    Compute face box extents of a numDims-dimensional patch whos extents
    are ilower,iupper. This is the face in the dim-dimension; side = -1
    means the left face, side = 1 the right face (so dim=1, side=-1 is the
    x-left face). Face extents are returned in faceLower, faceUpper
    _____________________________________________________________________*/
{
  faceLower = ilower;
  faceUpper = iupper;
  if (side < 0) {
    faceUpper[dim] = faceLower[dim];
  } else {
    faceLower[dim] = faceUpper[dim];
  }
  Print("Face(dim = %c, side = %d) box extents: ",dim+'x',side);
  printIndex(faceLower);
  fprintf(stderr," to ");
  printIndex(faceUpper);
  fprintf(stderr,"\n");
}

void IndexPlusPlus(const vector<int>& ilower,
                   const vector<int>& iupper,
                   const vector<bool>& active,
                   vector<int>& sub,
                   bool& eof)
  /*_____________________________________________________________________
    Function IndexPlusPlus
    Increment the d-dimensional subscript sub. This is useful when looping
    over a volume or an area. active is a d- boolean array. Indices with
    active=false are kept fixed, and those with active=true are updated.
    ilower,iupper specify the extents of the hypercube we loop over.
    eof is returned as 1 if we're at the end of the cube (the value of sub
    is set to ilower for active=1 indices; i.e. we do a periodic looping)
    and 0 otherwise.
    E.g., incrementing sub=(2,0,1) with active=(0,1,0), ilower=(0,0,0)
    and iupper=(2,2,2) results in sub=(0,0,2) and eof=1. If sub were
    (2,0,2) then sub=(0,0,0) and eof=1.
    _____________________________________________________________________*/
{
  assert((iupper.size() == ilower.size()) &&
         (sub.size()    == iupper.size()) &&
         (active.size() == sub.size()));

  int numDims = sub.size(), numDimsActive = 0, count = 0;
  for (int d = 0; d < numDims; d++) {
    if (active[d]) {
      numDimsActive++;
    }
  }
  eof = false;

  //  Print("BEFORE sub = ");
  //  printIndex(sub);
  //  fprintf(stderr,"\n");

  int d = 0;
  while ((!active[d]) && (d < numDims)) d++;
  if (d == numDims) {
    eof = true;
    return;
  }
  
  sub[d]++;
  if (sub[d] > iupper[d]) {
    while ((sub[d] > iupper[d]) || (!active[d])) {
      if (active[d]) sub[d] = ilower[d];
      d++;
      if (d == numDims) {
        eof = true;
        break;
      }
      if (active[d]) sub[d]++;
    }
  }

  //  Print("AFTER  sub = ");
  //  printIndex(sub);
  //  fprintf(stderr,"\n");
}

int
clean(void)
  /*_____________________________________________________________________
    Function clean:
    Exit MPI, debug modes. Call before each exit() call and in the end
    of the program.
    _____________________________________________________________________*/
{
#if DEBUG
  hypre_FinalizeMemoryDebug();
#endif
  MPI_Finalize();    // Quit MPI
}

static bool serializing = false;

void
serializeProcsBegin(void)
  /*_____________________________________________________________________
    Function serializeProcsBegin:
    Create a sequence of barriers to make sure that each proc separately
    goes thru the section following a call to serializeProcsBegin() 
    to it. I.e. that section of code is sequential - done first by proc 0,
    then proc 1, and so on. This is for better printout debugging with
    MPI.
    _____________________________________________________________________*/
{
  if (serializing) {
    fprintf(stderr,"\n\nError: serializeProcsBegin() called before "
            "serializeProcsEnd() done\n");
    clean();
    exit(1);
  }
  serializing = true;
#if DEBUG
  for (int i = 0; i < MYID; i++) {
    //    Print("serializeProcsBegin Barrier #%d\n",i);
    MPI_Barrier(MPI_COMM_WORLD); // Synchronize all procs to this point
  }
#endif
}

void
serializeProcsEnd(void)
  /*_____________________________________________________________________
    Function serializeProcsEnd:
    Create a sequence of barriers to make sure that each proc separately
    goes thru the section before a call to serializeProcsEnd() 
    to it. I.e. that section of code is sequential - done first by proc 0,
    then proc 1, and so on. This is for better printout debugging with
    MPI.
    _____________________________________________________________________*/
{
  static int numProcs = -1;
  if (numProcs == -1) {
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
  }
#if DEBUG
  for (int i = numProcs-1; i >= MYID; i--) {
    //    Print("serializeProcsEnd Barrier # %d\n",i);
    MPI_Barrier(MPI_COMM_WORLD); // Synchronize all procs to this point
  }
#endif
  if (!serializing) {
    fprintf(stderr,"\n\nError: serializeProcsEnd() called before "
            "serializeProcsBegin() done\n");
    clean();
    exit(1);
  }
  serializing = false;
}

void
pointwiseAdd(const vector<int>& x,
             const vector<int>& y,
             vector<int>& result)
{
  assert((x.size() == y.size()) &&
         (result.size() == y.size()));
  for (int d = 0; d < x.size(); d++) result[d] = x[d] + y[d];
}
void
pointwiseAdd(const vector<double>& x,
             const vector<double>& y,
             vector<double>& result)
{
  assert((x.size() == y.size()) &&
         (result.size() == y.size()));
  for (int d = 0; d < x.size(); d++) result[d] = x[d] + y[d];
}

void
scalarMult(const vector<double>& x,
           const double h,
           vector<double>& result)
{
  assert(result.size() == x.size());
  for (int d = 0; d < x.size(); d++) result[d] = h * x[d];
}

void
pointwiseMult(const vector<int>& i,
              const vector<int>& h,
              vector<int>& result)
{
  assert((i.size() == h.size()) &&
         (result.size() == h.size()));
  for (int d = 0; d < i.size(); d++) result[d] = i[d] * h[d];
}
void
pointwiseMult(const vector<int>& i,
              const vector<double>& h,
              vector<double>& result)
{
  assert((i.size() == h.size()) &&
         (result.size() == h.size()));
  for (int d = 0; d < i.size(); d++) result[d] = i[d] * h[d];
}

int
prod(const vector<int>& x)
{
  int result = 1;
  for (int d = 0; d < x.size(); d++) result *= x[d];
  return result;
}

double
prod(const vector<double>& x)
{
  double result = 1.0;
  for (int d = 0; d < x.size(); d++) result *= x[d];
  return result;
}

double
roundDigit(const double& x,
           const int d = 0)
  /*_____________________________________________________________________
    Function roundDigit:
    Round towards nearest rational (to certain # of digits).
    roundDigit(x,d) rounds the elements of x to the nearest rational
    a/(10^d * b), where a and b are integers.
    Alternatively, roundd(x,d) = 10^(-d) * round(10^d * x).
    roundDigit(x) is the same as roundDigit(x,0).
     _____________________________________________________________________*/
{

  double f = pow(10.0,d);
  return round(x*f)/f;
}

IntMatrix
grayCode(const int n,
         const vector<int> k)
  /*_____________________________________________________________________
    Function grayCode:
    This function returns a variable-base multiple-digit gray code.
    G = grayCode(N,K) returns the gray code permutation of the integers
    from 0 to prod(K)-1. N bust be a non-negative integer and K must be an
    N-vector of non-negative integers of bases. K[0] is the base of the
    right-most digit (LSB) in the N-digit string space, K[1] the base of
    the next right digit, and so on.
    The generated gray code is not necssarily cyclic. G is an array of size
    prod(K)xN, whose rows are the gray-code-ordered N-digit strings.
    _____________________________________________________________________*/
{
  assert(n > 0);
  assert(k.size() == n);
  int verbose = 0;
  
  int numRows = 1;
  int numCols = 0;
  int m = 0;
  numRows *= k[m];
  numCols++;
  IntMatrix G(numRows,numCols);
  for (int i = 0; i < k[0]; i++) {
    G(i,m) = i;
  }
  if (verbose >= 1) {
    Print("G for %d digits = \n",m);
    G.print(cout);
  }

  /* Generate G recursively */
  for (int m = 1; m < n; m++) {
    int b = k[m];
    numRows *= b;
    numCols++;
    IntMatrix Gnew(numRows,numCols);
    int startRow = 0;
    if (verbose >= 1) {
      Print("m = %d, b = %d\n",m,b);
    }
    for (int d = 0; d < b; d++) {
      if (d % 2) {           // d odd
        if (verbose >= 1) {
          Print("  (G*)^(%d)\n",d);
        }
        
        for (int i = 0; i < G.numRows(); i++) {
          for (int j = 0; j < G.numCols(); j++) {
            Gnew(startRow+G.numRows()-1-i,j) = G(i,j);
          }
          Gnew(startRow+G.numRows()-1-i,m) = d;
        }
        
      } else {               // d even
        if (verbose >= 1) {
          Print("  G^(%d)\n",d);
        }

        for (int i = 0; i < G.numRows(); i++) {
          for (int j = 0; j < G.numCols(); j++) {
            Gnew(startRow+i,j) = G(i,j);
          }
          Gnew(startRow+i,m) = d;
        }
      } // end d even
      startRow += G.numRows();
    } // end for d

    G = Gnew;
    if (verbose >= 1) {
      Print("G for %d digits = \n",m);
      G.print(cout);
    }
  } // end for m
  
  /* Check result */
  bool fail = false;
  for (int i = 0; i < G.numRows()-1; i++) {
    int diff = 0;
    for (int j = 0; j < G.numCols(); j++) {
      diff += int(abs(G(i,j) - G(i+1,j)));
    }
    if (diff != 1) {
      fail = true;
      Print("failed in difference between rows %d and %d\n",i,i+1);
      break;
    }
  } // end for i

  for (int i = 0; (i < G.numRows()) && (!fail); i++) {
    for (int i2 = 0; i2 < G.numRows(); i2++) {
      if (i != i2) {
        int diff = 0;
        for (int j = 0; j < G.numCols(); j++) {
          diff += int(abs(G(i,j) - G(i2,j)));
        }
        if (diff == 0) {
          fail = true;
          Print("failed in equality of rows %d and %d\n",i,i2);
          break;
        }
      }
    } // end for i2
  } // end for i

  if (fail) {
    Print("Gray code is incorrect!!!\n");
  } else {
    if (verbose >= 1) {
      Print("Gray code is correct.\n");
    }
  }

  return G;
} // end graycode
