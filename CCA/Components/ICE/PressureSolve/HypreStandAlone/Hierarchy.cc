#include "Hierarchy.h"
#include "util.h"
#include "Level.h"
#include "Patch.h"

#include <mpi.h>

/*_____________________________________________________________________
  Function makeHierarchy:
  Create a static refinement hierarchy hier into our data structures of
  Hierarchy, Levels and Patches. We define here all the patches owned
  by this proc only. Initialize the plevel and refinementRatio arrays
  needed by Hypre FAC. p = strcture of input parameters.
  _____________________________________________________________________*/

void
Hierarchy::make()
{
  // Hardcoded for a rectangular domain consisting of one box.
  // Dirichlet B.C. on the boundaries of the box
  // 2 levels, level 1 is twice finer and extends over the central half
  // of the domain.
  // TODO: restore domain as a union of boxes and add code to generally 
  // parse it and assign them to processors bla bla bla
  //  assert(_param->domain->patchList.size() == 1);

  const Counter numDims   = _param->numDims;
  const Counter numLevels = _param->numLevels;
  const Counter n         = _param->baseResolution;
//_param->domain->_resolution[0]; // Assumed uniform in all dimensions

  vector<Counter> k(numDims,2);
  IntMatrix procMap = grayCode(numDims,k);
  /* Print procMap */
  for (Counter i = 0;i < procMap.numRows();i++){
    for(Counter j = 0; j < procMap.numCols(); j++){
      if (j == 0) {
        Proc0Print("\t%d",procMap(i,j));
      } else {
        if (MYID == 0) {
          PrintNP("\t%d",procMap(i,j));
        }
      }
    }
    if (MYID == 0) {
      PrintNP("\n");
    }
  }

  /* Initialize the patches that THIS proc owns at all levels */
  for (Counter level = 0; level < numLevels; level++) {
    vector<Counter> refRat(numDims);
    /* Refinement ratio w.r.t. parent level. Assumed to be constant
       (1:2) in all dimensions and all levels for now. */
    if (level == 0) {        // Dummy ref. ratio value */
      for (Counter dim = 0; dim < numDims; dim++) {
        refRat[dim] = 1;
      }
    } else {
      for (Counter dim = 0; dim < numDims; dim++) {
        refRat[dim] = 2;
      }
    }

    double h;
    /* Compute meshsize, assumed the same in all directions at each level */
    if (level == 0) {
      h = 1.0/n;   // Size of domain divided by # of gridpoints
    } else {
      h = _levels[level-1]->_meshSize[0] / 
        refRat[0]; // ref. ratio constant for all dims
    }
    
    _levels.push_back(new Level(numDims,h));
    Level* lev = _levels[level];
    lev->_refRat = refRat;
    vector<int> ilower(numDims);
    vector<int> iupper(numDims);

    /* Mesh box extents (lower left corner, upper right corner) */
    for (Counter dim = 0; dim < numDims; dim++) {
      if( level == 0 ) {
        ilower[dim] = procMap(MYID,dim) * n/2;
        iupper[dim] = ilower[dim] + n/2 - 1;
      } else if( level == 1 ) {
        ilower[dim] = n/2 + procMap(MYID,dim) * n/2;
        iupper[dim] = ilower[dim] + n/2 - 1;
      } else {
        PrintNP("Unknown level\n");
        clean();
        exit(1);
      }
    }
    Patch* patch = new Patch(MYID,level,ilower,iupper);
    /* Figure out whether you are next to the domain boundary and set
       boundary condition there. */
    for (Counter d = 0; d < numDims; d++) {
      Side s = Left;
      if (ilower[d] == 0) {
        patch->setBoundaryType(d,s,Patch::Domain);
        patch->setBC(d,s,Patch::Dirichlet); // Hard coded to Dirichlet B.C.
      } else {
        patch->setBoundaryType(d,s,Patch::CoarseFine);
        patch->setBC(d,s,Patch::NA);
      }
      
      s = Right;
      if (iupper[d] == lev->_resolution[d]-1) {
        patch->setBoundaryType(d,s,Patch::Domain);
        patch->setBC(d,s,Patch::Dirichlet); // Hard coded to Dirichlet B.C.
      } else {
        patch->setBoundaryType(d,s,Patch::CoarseFine);
        patch->setBC(d,s,Patch::NA);
      }
    }

    lev->_patchList.push_back(patch);

  } // end for level

  getPatchesFromOtherProcs();  // Compute global patch IDs
}


void
Hierarchy::getPatchesFromOtherProcs()
{
  /* Types for arrays that are sent/received through MPI are int;
     convert to Counter later on, but don't risk having MPI_INT and
     Counter mixed up in the same call. */
  int sendbuf[_param->numProcs];  /* Suppose this proc is #0 has 5 patches,
                                   don't know what everyone else has.
                                   So this array will look like : 5 0 0 0 0. */
  int numPatches[_param->numProcs];  // After assembling sendbuf from all procs

  for(Counter level = 0; level < _levels.size(); level++ ) {
    Level* lev = _levels[level];
    //    const vector<Counter>& resolution = lev->_resolution;
    // clear sendbuf
    for( int index = 0; index < _param->numProcs; index++ ) {
      if( index == MYID ) {
        sendbuf[index] = lev->_patchList.size();
      } else {
        sendbuf[index] = 0;
      }
    }

    // Talk to all procs to find out how many patches they have on
    // this level.
    MPI_Allreduce(sendbuf, numPatches, _param->numProcs,
                  MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    int totalPatches = 0;
    int startPosition = 0;
    for(int index = 0; index < _param->numProcs; index++ ) {
      //      Proc0Print("has %d patches on level %d\n",
      // numPatches[index], level);
      totalPatches += numPatches[index];
      if( index < MYID )
        startPosition += numPatches[index];
    }

    //    Proc0Print("got totalPatches of %d\n", totalPatches);

    // Put our patch information into a big vector, share it with
    // all other procs
    int recordSize = 2*_param->numDims+1;
    int * sendPatchInfo = new int[ recordSize * totalPatches ];
    int * patchInfo     = new int[ recordSize * totalPatches ];
    for( int index = 0; index < recordSize*totalPatches; index++ ) {
      sendPatchInfo[index] = 0;
      patchInfo[index] = -1;
    }
    int count = startPosition * recordSize;
    for(Counter index = 0; index < lev->_patchList.size(); index++ ) {
      Patch* patch = lev->_patchList[index];
      sendPatchInfo[count++] = patch->_procID;
      for (Counter d = 0; d < _param->numDims; d++)
        sendPatchInfo[count++] = patch->_ilower[d];
      for (Counter d = 0; d < _param->numDims; d++)
        sendPatchInfo[count++] = patch->_iupper[d];
    }

    MPI_Allreduce(sendPatchInfo, patchInfo, totalPatches*recordSize,
                  MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    
    //    for(int index = 0; index < recordSize*totalPatches; index++ ) {
    //Proc0Print("%3d: %3d %3d\n",index,sendPatchInfo[index],patchInfo[index]);
    //}
    //    Print("Looping over patches and setting their boundary types\n");
    vector<int> ilower(_param->numDims);
    vector<int> iupper(_param->numDims);
    vector<int> otherilower(_param->numDims);
    vector<int> otheriupper(_param->numDims);
    int patchIndex = 0;

    for (Counter index = 0; index < totalPatches; index++ ) {
      int owner = patchInfo[recordSize*index];
      if (MYID != owner) {   // This patch is processed only on its owning proc
        continue;
      }
      Patch* patch = lev->_patchList[patchIndex];
      patchIndex++;
      for (Counter d = 0; d < _param->numDims; d++) {
        ilower[d] = patchInfo[recordSize*index + d + 1];
        iupper[d] = patchInfo[recordSize*index + _param->numDims + d + 1];

        /* Defaults: boundary is a C/F boundary; boundary condition is
           not applicable. */
        for (Side s = Left; s <= Right; ++s) {
          if (patch->getBoundaryType(d,s) != Patch::Domain) {
            patch->setBoundaryType(d,s,Patch::CoarseFine);
            patch->setBC(d,s,Patch::NA);
          }
        }

        /* Check if patch is near a domain boundary */
        // Hardcoded to one domain box of size [0,0] to [resolution].
        // TODO: L-shaped domain with its own makeHierarchy() function of
        // patches. Right now hard-coded to two levels and 2^d processors.
      }

      for (Counter other = 0; other < totalPatches; other++) {
        if (other == index)
          continue;
        for (Counter d = 0; d < _param->numDims; d++) {
          otherilower[d] = patchInfo[recordSize*other + d + 1];
          otheriupper[d] = patchInfo[recordSize*other + _param->numDims + d + 1];
        }
        /*
          Print("Comparing patch index=%d: from ",index);
          printIndex(ilower);
          PrintNP(" to ");
          printIndex(iupper);
          PrintNP("  owned by proc %d",patchInfo[recordSize*index]);
          PrintNP("\n");
          Print("To patch other=%d: from ",other);
          printIndex(otherilower);
          PrintNP(" to ");
          printIndex(otheriupper);
          PrintNP("  owned by proc %d",patchInfo[recordSize*other]);
          PrintNP("\n");
        */
        for (Counter d = 0; d < _param->numDims; d++) {
          /* Check if patch has a nbhring patch on its left */
          if (ilower[d] == otheriupper[d]+1) {
            for (Counter d2 = 0; d2 < _param->numDims; d2++) {
              if (d2 == d) continue;
              if (max(ilower[d2],otherilower[d2]) <=
                  min(iupper[d2],otheriupper[d2])) {
                patch->setBoundaryType(d,Left,Patch::Neighbor);
              }
            }
          }

          /* Check if patch has a nbhring patch on its right */
          if (iupper[d] == otherilower[d]-1) {
            for (Counter d2 = 0; d2 < _param->numDims; d2++) {
              if (d2 == d) continue;
              if (max(ilower[d2],otherilower[d2]) <=
                  min(iupper[d2],otheriupper[d2])) {
                patch->setBoundaryType(d,Right,Patch::Neighbor);
              }
            }
          }
        }
      } // end for other
    } // end for index
    delete [] sendPatchInfo;
    delete [] patchInfo;
  } // end for level
} // getPatchesFromOtherProcs()

void
Hierarchy::printPatchBoundaries()
{
  serializeProcsBegin();
  /* Print boundary types */
  for(Counter level = 0; level < _levels.size(); level++ ) {
    Print("---- Patch boundaries at level %d ----\n",level);
    Level* lev = _levels[level];
    for (Counter index = 0; index < lev->_patchList.size(); index++ ) {
      Patch* patch = lev->_patchList[index];
      Print("Patch # %d: from ",index);
      printIndex(patch->_ilower);
      PrintNP(" to ");
      printIndex(patch->_iupper);
      PrintNP("\n");
      for (Counter d = 0; d < _param->numDims; d++) {
        for (Side s = Left; s <= Right; ++s) {
          Print("  boundary( d = %d , s = %+d ) = %s\n",
                d,s,
                Patch::boundaryTypeString[patch->getBoundaryType(d,s)].c_str());
        }
      }
    }
  } // end for level
  Print("\n");
  serializeProcsEnd();
} // end getPatchesFromOtherProcs()
