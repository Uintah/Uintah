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
          const Counter numDims)
  /*_____________________________________________________________________
    Function copyIndex:
    Copy an index subFrom to index subTo in numDims-dimensions.
    _____________________________________________________________________*/
{
  for (Counter d = 0; d < numDims; d++) (*subTo)[d] = subFrom[d];
}

void 
ToIndex(const vector<Counter>& subFrom,
        Index* subTo,
        const Counter numDims)
  /*_____________________________________________________________________
    Function ToIndex:
    Convert a vector<Counter>-type subscript "subFrom" to Index-type "subTo",
    so that we can enjoy the flexibility of vector, but interface to
    Hypre with the correct type (Index). Note that Index is an int so we
    can make the conversion.
    _____________________________________________________________________*/
{
  assert(subFrom.size() == numDims);
  for (Counter d = 0; d < numDims; d++) (*subTo)[d] = int(subFrom[d]);
}

void 
ToIndex(const vector<int>& subFrom,
        Index* subTo,
        const Counter numDims)
  /*_____________________________________________________________________
    Function ToIndex:
    Convert a vector<int>-type subscript "subFrom" to Index-type "subTo",
    so that we can enjoy the flexibility of vector, but interface to
    Hypre with the correct type (Index).
    _____________________________________________________________________*/
{
  assert(subFrom.size() == numDims);
  for (Counter d = 0; d < numDims; d++) (*subTo)[d] = subFrom[d];
}

void 
Proc0Print(char *fmt, ...)
{
  //#if DRIVER_DEBUG
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
  //#endif
}

void 
Print(char *fmt, ...)
  /*_____________________________________________________________________
    Function Print:
    Print an output line on the current processor. Useful to parse MPI
    output.
    _____________________________________________________________________*/
{
#if DRIVER_DEBUG
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
#endif
}

void 
PrintNP(char *fmt, ...)
  /*_____________________________________________________________________
    Function Print:
    Print an output line on the current processor. Useful to parse MPI
    output.
    _____________________________________________________________________*/
{
#if DRIVER_DEBUG
  int vb = 1; /* Verbose level */
  va_list ap;
  va_start(ap, fmt);
  if (vb) {
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
#endif
}

template<class T>
void 
printIndex(const vector<T>& sub) 
{
  /*_____________________________________________________________________
    Function printIndex:
    Print vector-type numDims-dimensional index sub
    _____________________________________________________________________*/
#if DRIVER_DEBUG
  PrintNP("[");
  for (Counter d = 0; d < sub.size(); d++) {
    PrintNP("%d",sub[d]);
    if (d < sub.size()-1) PrintNP(",");
  }
  PrintNP("]");
#endif
}

template<>
void 
printIndex<double>(const vector<double>& sub) 
{
  /*_____________________________________________________________________
    Function printIndex:
    Print vector<double>-type numDims-dimensional index sub
    _____________________________________________________________________*/
#if DRIVER_DEBUG
  PrintNP("[");
  for (Counter d = 0; d < sub.size(); d++) {
    PrintNP("%.3f",sub[d]);
    if (d < sub.size()-1) PrintNP(",");
  }
  PrintNP("]");
#endif
}

void
faceExtents(const vector<int>& ilower,
            const vector<int>& iupper,
            const Counter d,
            const int s,
            vector<int>& faceLower,
            vector<int>& faceUpper)
  /*_____________________________________________________________________
    Function faceExtents:
    Compute face box extents of a numDims-dimensional patch whos extents
    are ilower,iupper. This is the face in the d-dimension; s = Left
    means the left face, s = Right the right face (so d=1, s=Left is the
    x-left face). Face extents are returned in faceLower, faceUpper
    _____________________________________________________________________*/
{
  faceLower = ilower;
  faceUpper = iupper;
  if (s == Left) {
    faceUpper[d] = faceLower[d];
  } else {
    faceLower[d] = faceUpper[d];
  }
#if DRIVER_DEBUG
  Print("Face(d = %c, s = %s) box extents: ",
        d+'x',(s == Left) ? "Left" : "Right");
  printIndex(faceLower);
  PrintNP(" to ");
  printIndex(faceUpper);
  PrintNP("\n");
#endif
}

Counter
numCellsInBox(const vector<int>& x,
              const vector<int>& y)
  /*_____________________________________________________________________
    Function numCellsInBox:
    Computes the total number of cells in the box [x,y]. x is the lower
    left corner and y is the upper right corner (in d-dimensions). This
    includes x and y (so like: prod(y-x+1)).
     _____________________________________________________________________*/
{
  Counter result = 1;
  for (Counter d = 0; d < x.size(); d++) {
    int temp = y[d] - x[d] + 1;
    if (temp < 0)
      return 0;
    result *= Counter(temp);
  }
  return result;
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

  Counter numDims = sub.size(), numDimsActive = 0;
  for (Counter d = 0; d < numDims; d++) {
    if (active[d]) {
      numDimsActive++;
    }
  }
  eof = false;

  Counter d = 0;
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
}

int
clean(void)
  /*_____________________________________________________________________
    Function clean:
    Exit MPI, debug modes. Call before each exit() call and in the end
    of the program.
    _____________________________________________________________________*/
{
  Print("Cleaning\n");
#if DRIVER_DEBUG
  hypre_FinalizeMemoryDebug();
#endif
  MPI_Finalize();    // Quit MPI
  return 0;
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
#if DRIVER_DEBUG
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
#if DRIVER_DEBUG
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

template<class T, class S>
void
pointwiseAdd(const vector<T>& x,
             const vector<S>& y,
             vector<S>& result)
{
  assert((x.size() == y.size()) &&
         (result.size() == y.size()));
  for (Counter d = 0; d < x.size(); d++) result[d] = x[d] + y[d];
}

template<class T>
void
scalarMult(const vector<T>& x,
           const T& h,
           vector<T>& result)
{
  assert(result.size() == x.size());
  for (Counter d = 0; d < x.size(); d++) result[d] = h * x[d];
}

template<class T, class S>
void
pointwiseMult(const vector<T>& i,
              const vector<S>& h,
              vector<S>& result)
{
  assert((i.size() == h.size()) &&
         (result.size() == h.size()));
  for (Counter d = 0; d < i.size(); d++) result[d] = i[d] * h[d];
}

template<class T, class S>
void
pointwiseDivide(const vector<S>& x,
                const vector<T>& y,
                vector<S>& result)
{
  assert((x.size() == y.size()) &&
         (result.size() == y.size()));
  for (Counter d = 0; d < x.size(); d++) result[d] = x[d] / y[d];
}

template<class T>
T
prod(const vector<T>& x)
{
  T result = 1;
  for (Counter d = 0; d < x.size(); d++) result *= x[d];
  return result;
}

double
roundDigit(const double& x,
           const Counter d /* = 0 */)
  /*_____________________________________________________________________
    Function roundDigit:
    Round towards nearest rational (to certain # of digits).
    roundDigit(x,d) rounds the elements of x to the nearest rational
    a/(10^d * b), where a and b are integers.
    Alternatively, roundd(x,d) = 10^(-d) * round(10^d * x).
    roundDigit(x) is the same as roundDigit(x,0).
     _____________________________________________________________________*/
{

  double f = pow(10.0,1.0*d);
  return round(x*f)/f;
}

IntMatrix
grayCode(const Counter n,
         const vector<Counter>& k)
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
  
  Counter numRows = 1;
  Counter numCols = 0;
  Counter m = 0;
  numRows *= k[m];
  numCols++;
  IntMatrix G(numRows,numCols);
  for (Counter i = 0; i < k[0]; i++) {
    G(i,m) = i;
  }
  if (verbose >= 1) {
    Print("G for %d digits = \n",m);
    G.print(cout);
  }

  /* Generate G recursively */
  for (Counter m = 1; m < n; m++) {
    Counter b = k[m];
    numRows *= b;
    numCols++;
    IntMatrix Gnew(numRows,numCols);
    Counter startRow = 0;
    if (verbose >= 1) {
      Print("m = %d, b = %d\n",m,b);
    }
    for (Counter d = 0; d < b; d++) {
      if (d % 2) {           // d odd
        if (verbose >= 1) {
          Print("  (G*)^(%d)\n",d);
        }
        
        for (Counter i = 0; i < G.numRows(); i++) {
          for (Counter j = 0; j < G.numCols(); j++) {
            Gnew(startRow+G.numRows()-1-i,j) = G(i,j);
          }
          Gnew(startRow+G.numRows()-1-i,m) = d;
        }
        
      } else {               // d even
        if (verbose >= 1) {
          Print("  G^(%d)\n",d);
        }

        for (Counter i = 0; i < G.numRows(); i++) {
          for (Counter j = 0; j < G.numCols(); j++) {
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
  for (Counter i = 0; i < G.numRows()-1; i++) {
    Counter diff = 0;
    for (Counter j = 0; j < G.numCols(); j++) {
      diff += Counter(abs(G(i,j) - G(i+1,j)));
    }
    if (diff != 1) {
      fail = true;
      Print("failed in difference between rows %d and %d\n",i,i+1);
      break;
    }
  } // end for i

  for (Counter i = 0; (i < G.numRows()) && (!fail); i++) {
    for (Counter i2 = 0; i2 < G.numRows(); i2++) {
      if (i != i2) {
        Counter diff = 0;
        for (Counter j = 0; j < G.numCols(); j++) {
          diff += Counter(abs(G(i,j) - G(i2,j)));
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

// The compiler was not instantiating these templated functions, so
// this forces it to do so.
void
forceInstantiation()
{
  vector<Counter> c;
  vector<double>  d;
  vector<int>     i;

  pointwiseMult( c, i, i );
  pointwiseMult( i, d, d );
  pointwiseAdd( i, i, i );
  pointwiseAdd( d, d, d );
  pointwiseDivide( c, i, c );
  pointwiseDivide( i, c, i );
  scalarMult( d, 1.0, d );
  prod( d );
  prod( i );
  printIndex( c );
  printIndex( d );
  printIndex( i );
}
