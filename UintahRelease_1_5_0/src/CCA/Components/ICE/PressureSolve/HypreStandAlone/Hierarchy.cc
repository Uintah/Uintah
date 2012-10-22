/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include "Hierarchy.h"
#include "util.h"
#include "Level.h"
#include "Patch.h"
#include <sci_defs/mpi_defs.h>
using namespace std;

void
Hierarchy::make()
  /*_____________________________________________________________________
    Function Hierarchy::make()
    Create a static refinement hierarchy hier into our data structures of
    Hierarchy, Levels and Patches. We define all patches owned by ALL
    processors, not only the ones owned by this proc. 
    We also initialize the plevel and refinementRatio arrays
    needed by Hypre FAC.
    _param = strcture of input parameters.
    _____________________________________________________________________*/
{
  /* Hardcoded for a rectangular domain consisting of one box.
     Dirichlet B.C. on the boundaries of the box 2 levels, level 1 is
     twice finer and extends over the central half of the domain.

     TODO: restore domain as a union of boxes and add code to generally
     parse it and assign them to processors bla bla bla
     assert(_param->domain->patchList.size() == 1);
  */

  serializeProcsBegin();
  const Counter numDims   = _param->numDims;
  const Counter numLevels = _param->numLevels;
  const Counter n         = _param->baseResolution;
  const int numProcs      = _param->numProcs;
  //_param->domain->_resolution[0]; // Assumed uniform in all dimensions

  /* Initialize the patches that THIS proc owns at all levels */
  for (Counter level = 0; level < numLevels; level++) {
    Vector<Counter> k(0,numDims,0,"k",2);
    IntMatrix procMap = grayCode(numDims,k);
    const Counter& numPatches = procMap.numRows();

    /* Print procMap */
    dbg0 << "procMap at level " << level << "\n";
    dbg0 << procMap << "\n";

    dbg.setLevel(10);
    dbg << "Setting refinement ratio" << "\n";

    /* Set Refinement ratio w.r.t. parent level. Hard-coded to be
       constant (1:2) in all dimensions and all levels for now. */
    Vector<Counter> refRat(0,numDims);
    if (level == 0) {        /* Dummy ref. ratio value */
      for (Counter dim = 0; dim < numDims; dim++) {
        refRat[dim] = 1;
      }
    } else {
      for (Counter dim = 0; dim < numDims; dim++) {
        refRat[dim] = 2;
      }
    }

    dbg.setLevel(10);
    dbg << "Setting meshsize" << "\n";
    double h;
    /* Compute meshsize, assumed the same in all directions at each level */
    if (level == 0) {
      h = 1.0/n;   // Size of domain divided by # of gridpoints
    } else {
      h = _levels[level-1]->_meshSize[0] /  refRat[0]; /* ref. ratio
                                                          assumed constant for
                                                          all dims */
    }
    
    dbg.setLevel(10);
    dbg << "Initializing Level object and adding it to hier:" << "\n";
    _levels.push_back(new Level(numDims,h));
    Level* lev = _levels[level];
    lev->_refRat = refRat;
    lev->_patchList.resize(numProcs);
    int offset = level; // For tests where the fine patch is owned by
    // one proc, and its parent patch is owned by another proc

    /* Mesh box extents (lower left corner, upper right corner) */
    /* This is where the specific geometry of the hierarchy is
       hard-coded. */
    dbg.setLevel(2);
    dbg << "numPatches = " << numPatches << "\n";
    dbg.indent();
    for (Counter i = 0; i < numPatches; i++) {
      int owner = (i+offset) % numProcs; // Owner of patch i, hard-coded
      dbg.setLevel(2);
      dbg << "Creating Patch i = " << setw(2) << right << i
          << " owned by proc " << setw(2) << right << owner << "\n";
      dbg.indent();
      Vector<int> lower(0,numDims);
      Vector<int> upper(0,numDims);
      for (Counter dim = 0; dim < numDims; dim++) {
        switch (level) {
        case 0:
          {
            lower[dim] = procMap(i,dim) * n/2;
            upper[dim] = lower[dim] + n/2 - 1;
            break;
          }
        case 1:
          {
            lower[dim] = n/2 + procMap(i,dim) * n/2;
            upper[dim] = lower[dim] + n/2 - 1;
            break;
          }
        default:
          {
            cerr << "\n\nError, unsupported level=" << level 
                 << " in Hierarchy::make()" << "\n";
            clean();
            exit(1);
          }
        } // end switch (level)
      } // for dim
      
      dbg.setLevel(10);
      dbg << " lower = " << lower
          << " upper = " << upper
          << "\n";
      // Create patch object from the geometry parameters above and
      // add it to the level's appropriate list of patches.
      Patch* patch = scinew Patch(owner,level,Box(lower,upper));
      dbg.setLevel(10);
      dbg << "box = " << patch->_box << "\n"
          << "box.get(Left ) = " << patch->_box.get(Left) << "\n"
          << "box.get(Right) = " << patch->_box.get(Right) << "\n";
      patch->setDomainBoundaries(*lev);
      lev->_patchList[owner].push_back(patch);
      dbg.unindent();
    } // end for i (patches)
    dbg.unindent();
    dbg.setLevel(2);
    dbg << "\n";
  } // end for level

  serializeProcsEnd();

  getPatchesFromOtherProcs();  // Compute global patch IDs
}


void
Hierarchy::getPatchesFromOtherProcs(void)
  //_____________________________________________________________________
  // Function Hierarchy::getPatchesFromOtherProcs()
  // Assemble all processors' data on patches and generate global patch IDs
  // across all levels and owning procs. We do not seem patch IDs right now.
  // In addition, send boundary information for all patches. Again, as
  // all procs know about all patches now, we don't need MPI_AllReduce().
  //_____________________________________________________________________
{
  serializeProcsBegin();
  funcPrint("Hierarchy::getPatchesFromOtherProcs()",FBegin);
  Counter globalPatchID = 0;
  for (Counter level = 0; level < _levels.size(); level++) {
    if (level > 0) dbg0 << "\n";
    dbg0 << "Computing global patch IDs & BC info at level = "
         << level << "\n";
    dbg.setLevel(2);
    dbg << "Computing global patch IDs & BC info at level = "
         << level << "\n";
    dbg.indent();
    Level* lev = _levels[level];
    for (Counter owner = 0; owner < lev->_patchList.size(); owner++) {
      dbg.setLevel(2);
      dbg << "==== Owner = " << owner << " ====" << "\n";
      dbg.indent();
      for (Counter index = 0; index < lev->_patchList[owner].size();index++) {
        Patch* patch = lev->_patchList[owner][index];
        const Box& box = patch->_box;

        // Set global patch ID
        patch->_patchID = globalPatchID;
        dbg.setLevel(10);
        dbg << "Updated patch:" << "\n";
        dbg << *patch << "\n";
        globalPatchID++;
        
        // Set defaults internal boundaries to C/F, bc = not applicable.
        for (Counter d = 0; d < _param->numDims; d++) {
          for (Side s = Left; s <= Right; ++s) {
            if (patch->getBoundaryType(d,s) != Patch::Domain) {
              patch->setBoundaryType(d,s,Patch::CoarseFine);
              patch->setBC(d,s,Patch::NA);
            }
          }
        }

        // Check for nbhring patches of this patch at the same level
        dbg.indent();
        for (Counter owner2 = 0; owner2 < lev->_patchList.size(); owner2++) {
          for (Counter index2 = 0; index2 < lev->_patchList[owner2].size();
               index2++) {
            Patch* otherPatch = lev->_patchList[owner2][index2];
            if (otherPatch == patch) {
              continue;
            }
            dbg.setLevel(3);
            dbg << "==== Comparing ====" << "\n";
            dbg << "Patch = " << "\n";
            dbg << *patch << "\n";
            dbg << "otherPatch = " << "\n";
            dbg << *otherPatch << "\n";
            const Box& otherBox = otherPatch->_box;
            const Box& intersectBox = box.intersect(otherBox);
            for (Counter d = 0; d < _param->numDims; d++) {
              dbg.setLevel(3);
              dbg << "Looping, d = " << d << "\n";
              for (Side s = Left; s <= Right; ++s) {
                dbg.setLevel(3);
                dbg << "Looping, s = " << s
                    << "  box.get(s)[d] + int(s) = " << box.get(s)[d] + int(s)
                    << "  otherBox.get(-s)[d]     = "
                    << otherBox.get(Side(-s))[d]
                    << "\n";
                // Check if otherPatch is a nbhr on side "side" of patch
                if (box.get(s)[d] + int(s) == otherBox.get(Side(-s))[d]) {
                  dbg.setLevel(3);
                  dbg << "otherBox nbhring box for (d = " << d
                      << " , s = " << s << ")" << "\n";
                  for (Counter d2 = 0; d2 < _param->numDims; d2++) {
                    if ((d2 != d) && (!intersectBox.degenerate(d2))) {
                      patch->setBoundaryType(d,s,Patch::Neighbor);
                      dbg.setLevel(2);
                      dbg << "Marking patch boundary (d = " << d
                          << " , s = " << s << ") to Neighbor" << "\n";
                      break; // Forgotten in the original code?
                    } // end if (d2 != d) && non-empty intersection
                  } // end for d2
                } // end if otherBox nbhring box in the (d,s) face
              } // end for s
            } // end for d
          } // end for index2 (other patches)
        } // end for owner2
        dbg.unindent();
      } // end for index (patches)
      dbg.unindent();
    } // end for owner
    dbg.unindent();
  } // end for level
  serializeProcsEnd();
  dbg0 << "\n";
  funcPrint("Hierarchy::getPatchesFromOtherProcs()",FEnd);
} // getPatchesFromOtherProcs()

void
Hierarchy::printPatchBoundaries()
{
  serializeProcsBegin();
  /* Print boundary types */
  dbg.setLevel(2);
  for(Counter level = 0; level < _levels.size(); level++ ) {
    dbg << "---- Patch boundaries at level " << level << " ----" << "\n";
    Level* lev = _levels[level];
    for (Counter index = 0; index < lev->_patchList[MYID].size(); index++ ) {
      Patch* patch = lev->_patchList[MYID][index];
      dbg << "Patch #" << patch->_patchID
          << ", owned by proc " << patch->_procID << ":" << "\n";
      dbg << patch->_box;
      dbg << "\n";
      for (Counter d = 0; d < _param->numDims; d++) {
        for (Side s = Left; s <= Right; ++s) {
          dbg << "  boundary(d = " << d
              << " , s = " << s << ") = "
              << patch->getBoundaryType(d,s) << "\n";
        }
      }
    }
  } // end for level
  dbg << "\n";
  serializeProcsEnd();
} // end printPatchBoundaries()

std::vector<Patch*>
Hierarchy::finePatchesOverMe(const Patch& patch) const
{
  std::vector<Patch*> finePatchList;
  const Counter& level = patch._levelID;
  const Counter& numLevels = _levels.size();
  const int numProcs      = _param->numProcs;

  if (level == numLevels) {
    // Finest level, nothing on top of me, return empty list
    return finePatchList;
  }

  const Counter fineLevel = level+1;
  dbg.setLevel(2);
  dbg << "======== finePatchesOverMe BEGIN ========" << "\n";
  dbg << "Searching level " << fineLevel << " patches above patch "
      << "ID=" << setw(2) << left << patch._patchID << " "
      << "owner=" << setw(2) << left << patch._procID << " "    
      << patch._box << "\n";
  const Vector<Counter>& refRat = _levels[fineLevel]->_refRat;
  Box coarseRefined(patch._box.get(Left) * refRat,
                    (patch._box.get(Right) + 1) * refRat - 1);
  dbg << "coarseRefined " << coarseRefined << "\n";
  dbg.indent();
  for (int owner = 0; owner < numProcs; owner++) {
    dbg.setLevel(3);
    dbg << "Looking in patch list of owner = " << owner << "\n";
    vector<Patch*>& ownerList = _levels[fineLevel]->_patchList[owner];
    for (vector<Patch*>::iterator iter = ownerList.begin();
         iter != ownerList.end(); ++iter) {
      Patch* finePatch = *iter;
      dbg.setLevel(3);
      dbg << "Considering patch "
          << "ID=" << setw(2) << left << finePatch->_patchID << " "
          << "owner=" << setw(2) << left << finePatch->_procID << " "
          << finePatch->_box << " ..." << "\n";
      if (!coarseRefined.intersect(finePatch->_box).degenerate()) {
        // Non-empty patch, finePatch intersection ==> add to list
        dbg.setLevel(2);
        dbg << "Found patch "
            << "ID=" << setw(2) << left << finePatch->_patchID << " "
            << "owner=" << setw(2) << left << finePatch->_procID << " "
            << finePatch->_box << "\n";
        finePatchList.push_back(*iter);
      }
    }
  }
  dbg.unindent();
  dbg.setLevel(2);
  dbg << "======== finePatchesOverMe END ========" << "\n";
  return finePatchList;
}

std::ostream&
operator << (std::ostream& os, const Hierarchy& hier)
  // Write the Hierarchy to the output stream os.
{
  for (Counter level = 0; level < hier._levels.size(); level++) {
    os << "---- Level " << level << " ----" << "\n";
    os << *(hier._levels[level]) << "\n";
  } // end for level
  return os;
} // end operator <<
