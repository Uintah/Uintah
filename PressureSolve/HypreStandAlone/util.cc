/*--------------------------------------------------------------------------
 * File: util.cc
 *
 * Utility functions: printouts, index conversion, cleanups.
 *
 * Revision history:
 * 20-JUL-2005   Oren   Created.
 *--------------------------------------------------------------------------*/
#include "mydriver.h"
#include <vector>
using namespace std;

void 
ToIndex(const vector<int>& subFrom,
        Index* subTo)
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
    printf("PROC %2d: ",MYID);
    vprintf(fmt, ap);
  }
  fflush(stdout);
  if (vb) {
    va_start(ap, fmt);
    //    if (log_file)
    //      vfprintf(log_file, fmt, ap);
    //    if (log_file)
    //      fflush(log_file);
  }
  va_end(ap);
}

void 
printIndex(const vector<int>& sub) 
{
  /*_____________________________________________________________________
    Function printIndex:
    Print vector-type numDims-dimensional index sub
    _____________________________________________________________________*/
  printf("[");
  for (int d = 0; d < sub.size(); d++) {
    printf("%d",sub[d]);
    if (d < sub.size()-1) printf(",");
  }
  printf("]");
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
  printf(" to ");
  printIndex(faceUpper);
  printf("\n");
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
  assert(iupper.size() == ilower.size());
  assert(sub.size()    == ilower.size());
  assert(sub.size()    == iupper.size());
  assert(active.size() == sub.size());
  assert(active.size() == ilower.size());
  assert(active.size() == iupper.size());

  int numDims = sub.size(), numDimsActive = 0, count = 0;
  for (int d = 0; d < numDims; d++) {
    if (active[d]) {
      numDimsActive++;
    }
  }
  eof = false;

  //  Print("BEFORE sub = ");
  //  printIndex(sub);
  //  printf("\n");

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
  //  printf("\n");
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
