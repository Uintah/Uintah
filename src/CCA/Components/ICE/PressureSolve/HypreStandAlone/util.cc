/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


/*--------------------------------------------------------------------------
 * File: util.cc
 *
 * Utility functions: printouts, index conversion, cleanups.
 *
 * Revision history:
 * 20-JUL-2005   Oren   Created.
 *--------------------------------------------------------------------------*/

#include "util.h"
#include "Vector.h"
#include "DebugStream.h"
#include <utilities.h>
using namespace std;

string lineHeader(const Counter indent)
  // Print header for each output line printed using DebugStream
{
  ostringstream os;
  os << "P"
     << setw(2) << left << MYID << ": ";
  for (Counter i = 0; i < 2*indent; i++) os << " ";
  return os.str();
}

void funcPrint(const string& name, const FuncPlace& p)
{
  string funcPlaceStr;
  if (p == FBegin) {
    funcPlaceStr = "begin";
  } else {
    funcPlaceStr = "end";
  }
  dbg.setLevel(3);
  dbg << name << " " << funcPlaceStr << "\n";  
}

void linePrint(const string& s, const Counter len)
{
  for (Counter i = 0; i < len; i++) {
    dbg0 << s;
  }
  dbg0 << "\n";
}

int
clean(void)
  /*_____________________________________________________________________
    Function clean:
    Exit MPI, debug modes. Call before each exit() call and in the end
    of the program.
    _____________________________________________________________________*/
{
  dbg.setLevel(10);
  dbg << "Cleaning" << "\n";
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
    cerr << "\n\nError: serializeProcsBegin() called before "
         << "serializeProcsEnd() done" << "\n";
    clean();
    exit(1);
  }
  serializing = true;
#if DRIVER_DEBUG
  for (int i = 0; i < MYID; i++) {
    //    dbg << "serializeProcsBegin Barrier "
    //        << setw(2) << right << i
    //        << "\n";
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
    //    dbg << "serializeProcsEnd   Barrier "
    //        << setw(2) << right << i
    //        << "\n";
    MPI_Barrier(MPI_COMM_WORLD); // Synchronize all procs to this point
  }
#endif
  if (!serializing) {
    cerr << "\n\nError: serializeProcsEnd() called before "
         << "serializeProcsBegin() done" << "\n";
    clean();
    exit(1);
  }
  serializing = false;
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
         const Vector<Counter>& k)
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
  assert(k.getLen() == n);
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
    dbg << "G for " << m << " digits =" << "\n";
    dbg << G;
  }

  /* Generate G recursively */
  for (Counter m = 1; m < n; m++) {
    Counter b = k[m];
    numRows *= b;
    numCols++;
    IntMatrix Gnew(numRows,numCols);
    Counter startRow = 0;
    if (verbose >= 1) {
      dbg << "m = " << m << " "
           << "b = " << b 
           << "\n";
    }
    for (Counter d = 0; d < b; d++) {
      if (d % 2) {           // d odd
        if (verbose >= 1) {
          dbg << "  (G*)^(" << d << ")" << "\n";
        }
        
        for (Counter i = 0; i < G.numRows(); i++) {
          for (Counter j = 0; j < G.numCols(); j++) {
            Gnew(startRow+G.numRows()-1-i,j) = G(i,j);
          }
          Gnew(startRow+G.numRows()-1-i,m) = d;
        }
        
      } else {               // d even
        if (verbose >= 1) {
          dbg << "  G^(" << d << ")" << "\n";
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
      dbg << "  G for " << m << " digits = " << G << "\n";
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
      dbg << "failed in difference between rows " << i
          << " and " << i+1 << "\n";
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
          dbg << "failed in equality of rows " << i
              << " and " << i2 << "\n";
          break;
        }
      }
    } // end for i2
  } // end for i

  if (fail) {
    dbg << "Gray code is incorrect!!!" << "\n";
  } else {
    if (verbose >= 1) {
      dbg << "Gray code is correct." << "\n";
    }
  }

  return G;
} // end graycode
