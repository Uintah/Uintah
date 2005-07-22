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
             Index* to,
             const int numDims);

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

void serializeProcsBegin(void);

void serializeProcsEnd(void);

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

/*================== Class delcarations ==================*/

class Param {
  /*_____________________________________________________________________
    class Param:
    a structure of input parameters.
    _____________________________________________________________________*/
public:
  int         numDims;      // # dimensions
  int         numLevels;    // # of levels in a static MR hierarchy
  int         numProcs;     // # of processors
  int         n;            // Resolution of Level 0 in all dimensions
  int         solverID;     // Solver ID, 30 = AMG, 99 = FAC
  bool        printSystem;  // Debugging flag for linear system file dump
};

class Patch {
  /*_____________________________________________________________________
    class Patch:
    A box of data at a certain level. A processor may own more than one
    patch. A patch can be owned by one proc only.
    _____________________________________________________________________*/
public:
  int         _procID;    // Owning processor's ID
  int         _levelID;   // Which level this Patch belongs to
  vector<int> _ilower;    // Lower left corner subscript
  vector<int> _iupper;    // Upper right corner subscript
  int         _numCells;  // Total # cells
  
  Patch(const int procID, 
        const int levelID,
        const vector<int>& ilower,
        const vector<int>& iupper)
  {
    _procID = procID;
    _levelID = levelID; 
    _ilower = ilower; 
    _iupper = iupper;
    _boundaries.resize(2*_ilower.size());
    vector<int> sz(_ilower.size());
    for (int d = 0; d < _ilower.size(); d++)
      sz[d] = _iupper[d] - _ilower[d] + 1;
    _numCells = prod(sz);
  }

  enum BoundaryType {
    Domain, CoarseFine, Neighbor
  };
  vector<BoundaryType> _boundaries;
  BoundaryType& getBoundary(int d, int s) {
    return _boundaries[2*d+(s+1)/2];
  }
private:
};

class Level {
  /*_____________________________________________________________________
    class Level:
    A union of boxes that share the same meshsize and index space. Each
    proc owns several boxes of a level, not necessarily the entire level.
    _____________________________________________________________________*/
public:
  int            _numDims;    // # dimensions
  vector<double> _meshSize;   // Meshsize in all dimensions
  vector<int>    _resolution; // Size(level) if extends over the full domain
  vector<Patch*> _patchList;  // owned by this proc ONLY

  Level(const int numDims,
        const double h) {
    /* Domain is assumed to be of size 1.0 x ... x 1.0 and
       1/h[d] is integer for all d. */
    _meshSize.resize(numDims);
    _resolution.resize(numDims);
    for (int d = 0; d < numDims; d++) {
      _meshSize[d]   = h;
      _resolution[d] = int(floor(1.0/_meshSize[d]));
    }
  }

private:
};

class Hierarchy {
  /*_____________________________________________________________________
    class Hierarchy:
    A sequence of Levels. Level 0 is the coarsest, Level 1 is the next
    finer level, and so on. Hierarchy is the same for all procs.
    _____________________________________________________________________*/
public:
  vector<Level*> _levels;
};
