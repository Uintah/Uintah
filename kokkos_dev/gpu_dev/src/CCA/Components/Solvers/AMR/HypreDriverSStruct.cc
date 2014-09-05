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
 * File: HypreDriverSStruct.cc
 *
 * Implementation of a wrapper of a Hypre solvers working with the Hypre
 * SStruct system interface.
 *--------------------------------------------------------------------------*/

#include <sci_defs/hypre_defs.h>

#include <CCA/Components/Solvers/AMR/HypreDriverSStruct.h>
#include <CCA/Components/Solvers/MatrixUtil.h>
#include <Core/Grid/AMR.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/Stencil7.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Parallel/Parallel.h>

#include <Core/Geometry/IntVector.h>
#include <Core/Math/MiscMath.h>
#include <Core/Util/DebugStream.h>

#include <iomanip>
#include <iostream>

using namespace Uintah;
using namespace std;

//#define DEBUG

//__________________________________
//  To turn on normal output
//  setenv SCI_DEBUG "HYPRE_DOING_COUT:+"

static DebugStream cout_doing("HYPRE_DOING_COUT", false);
static DebugStream cout_dbg("HYPRE_DBG", false);

//___________________________________________________________________
HypreDriverSStruct::HyprePatch::HyprePatch(const Patch* patch,
                                           const int matl) :
  _patch(patch), 
  _matl(matl),
  _level(patch->getLevel()->getIndex()),
  _low(patch->getCellLowIndex()),
  _high(patch->getCellHighIndex()-IntVector(1,1,1))
  // Note: we need to subtract (1,1,1) from high because our loops are
  // cell = low; cell <= high.
{
}

//___________________________________________________________________
// HypreDriverSStruct::HyprePatch bogus patch constructor
//   for when there are no patches on a level for a processor
//   this is a one celled "patch"
//___________________________________________________________________
HypreDriverSStruct::HyprePatch::HyprePatch(const int level,
                                           const int matl) :
  _patch(0), 
  _matl(matl),
  _level(level),
  _low(IntVector(Parallel::getMPIRank(),level,-9)),
  _high(IntVector(Parallel::getMPIRank(),level,-9))
{
}

HypreDriverSStruct::HyprePatch::~HyprePatch(void)
{}


//___________________________________________________________________
// HypreDriverSStruct destructor
//___________________________________________________________________
HypreDriverSStruct::~HypreDriverSStruct(void)
{
  // do nothing - should have been cleaned up after solve
}

void HypreDriverSStruct::cleanup(void)
{
  cout_doing << Parallel::getMPIRank() <<" HypreDriverSStruct cleanup\n";
  //printDataStatus();
  // Destroy matrix, RHS, solution objects
#if 1
  if (_exists[SStructA] >= SStructAssembled) {
    cout_doing << " Destroying A\n";
    HYPRE_SStructMatrixDestroy(_HA);
    _exists[SStructA] = SStructDestroyed;
  }
#endif
  if (_exists[SStructB] >= SStructAssembled) {
    HYPRE_SStructVectorDestroy(_HB);
    _exists[SStructB] = SStructDestroyed;
  }
  if (_exists[SStructX] >= SStructAssembled) {
    HYPRE_SStructVectorDestroy(_HX);
    _exists[SStructX] = SStructDestroyed;
  }
  // Destroy graph objects
  if (_exists[SStructGraph] >= SStructAssembled) {
    HYPRE_SStructGraphDestroy(_graph);
    _exists[SStructGraph] = SStructDestroyed;
  }

  // Destroying grid, stencil
  if (_exists[SStructStencil] >= SStructCreated) {
    HYPRE_SStructStencilDestroy(_stencil);
    _exists[SStructStencil] = SStructDestroyed;
  }
  if (_exists[SStructGrid] >= SStructAssembled) {
    HYPRE_SStructGridDestroy(_grid);
    _exists[SStructGrid] = SStructDestroyed;
  }
  if (_vars) {
    delete _vars;
    _vars = 0;
  }
}
//______________________________________________________________________
void
HypreDriverSStruct::printMatrix(const string& fileName )
{
  if (!_params->printSystem) return;
  HYPRE_SStructMatrixPrint((fileName + ".sstruct").c_str(), _HA, 0);
  if (_requiresPar) {
    HYPRE_ParCSRMatrixPrint(_HA_Par, (fileName + ".par").c_str());
    // Print CSR matrix in IJ format, base 1 for rows and cols
    //HYPRE_ParCSRMatrixPrintIJ(_HA_Par, 1, 1, (fileName + ".ij").c_str());
  }
}
//______________________________________________________________________
void
HypreDriverSStruct::printRHS(const string& fileName )
{
  if (!_params->printSystem) return;
  HYPRE_SStructVectorPrint(fileName.c_str(), _HB, 0);
  if (_requiresPar) {
    HYPRE_ParVectorPrint(_HB_Par, (fileName + ".par").c_str());
  }
}
//______________________________________________________________________
void
HypreDriverSStruct::printSolution(const string& fileName )
{
  if (!_params->printSystem) return;
  HYPRE_SStructVectorPrint(fileName.c_str(), _HX, 0);
  if (_requiresPar) {
    HYPRE_ParVectorPrint(_HX_Par, (fileName + ".par").c_str());
  }
}
//______________________________________________________________________
void
HypreDriverSStruct::gatherSolutionVector(void)
{
  HYPRE_SStructVectorGather(_HX);
} 
//______________________________________________________________________
//
void
HypreDriverSStruct::printDataStatus(void)
{
  cout_doing << "Hypre SStruct interface data status:" << "\n";
  for (unsigned i = 0; i < _exists.size(); i++) {
    cout_doing << "_exists[" << i << "] = " << _exists[i] << "\n";
  }
}

//#####################################################################
// class HypreDriverSStruct implementation for CC variable type
//#####################################################################

static const int CC_NUM_VARS = 1; // # Hypre var types that we use in CC solves
static const int CC_VAR = 0;      // Hypre CC variable type index


//___________________________________________________________________
// Function HypreDriverSStruct::makeLinearSystem_CC~
// Construct the linear system for CC variables (e.g. pressure),
// for the Hypre Struct interface (1-level problem / uniform grid).
// We set up the matrix at all patches of the "level" data member.
// matl is a fake material index. We always have one material here,
// matl=0 (pressure).
//  - 
//___________________________________________________________________
void
HypreDriverSStruct::makeLinearSystem_CC(const int matl)
{
  cout_doing << Parallel::getMPIRank() << "------------------------------ HypreDriverSStruct::makeLinearSystem_CC()" << "\n";
  ASSERTEQ(sizeof(Stencil7), 7*sizeof(double));
  //__________________________________
  // Set up the grid
  cout_doing << _pg->myrank() << " Setting up the grid" << "\n";
  // Create an empty grid in 3 dimensions with # parts = numLevels.
  const int numDims = 3;
  const int numLevels = _level->getGrid()->numLevels();
  HYPRE_SStructGridCreate(_pg->getComm(), numDims, numLevels, &_grid);

  _exists[SStructGrid] = SStructCreated;
  _vars = scinew HYPRE_SStructVariable[CC_NUM_VARS];
  _vars[CC_VAR] = HYPRE_SSTRUCT_VARIABLE_CELL; // We use only cell centered var

  // if my processor doesn't have patches on a given level, then we need to create
  // some bogus (irrelevent inexpensive) data so hypre doesn't crash.
  vector<bool> useBogusLevelData(numLevels, true);

  for (int p = 0 ; p < _patches->size(); p++) {
    HyprePatch_CC hpatch(_patches->get(p),matl);
    hpatch.addToGrid(_grid,_vars);
    useBogusLevelData[_patches->get(p)->getLevel()->getIndex()] = false;
  }
  
  for (int l = 0; l < numLevels; l++) {
    if (useBogusLevelData[l]) {
      HyprePatch_CC hpatch(l, matl);
      hpatch.addToGrid(_grid, _vars);
    }
  }

  delete _vars;
  _vars = 0;

  HYPRE_SStructGridAssemble(_grid);
  _exists[SStructGrid] = SStructAssembled;

  //==================================================================
  // Create the stencil
  //==================================================================
  if (_params->symmetric) { 
    // Match the ordering of stencil elements in Hypre and Stencil7. 
    // Ordering:                x- y- z- diagonal
    // Or in terms of Stencil7: w  s  b  p
    _stencilSize = numDims+1;
    int offsets[4][numDims] = {{-1,0,0},
                               {0,-1,0},
                               {0,0,-1},
                               {0,0,0}};

    HYPRE_SStructStencilCreate(numDims, _stencilSize, &_stencil);
    for (int i = 0; i < _stencilSize; i++) {
      HYPRE_SStructStencilSetEntry(_stencil, i, offsets[i], 0);
    }
  } else {
    // Ordering:                x- x+ y- y+ z- z+ diagonal
    // Or in terms of Stencil7: w  e  s  n  b  t  p
    _stencilSize = 2*numDims+1;
    int offsets[7][numDims] = {{-1,0,0}, {1,0,0},
                               {0,-1,0}, {0,1,0},
                               {0,0,-1}, {0,0,1},
                               {0,0,0}};

    HYPRE_SStructStencilCreate(numDims, _stencilSize, &_stencil);
    for (int i = 0; i < _stencilSize; i++) {
      HYPRE_SStructStencilSetEntry(_stencil, i, offsets[i], 0);
    }
  }
  _exists[SStructStencil] = SStructCreated;
  //==================================================================
  // Setup connection graph
  //================================================================== 
  cout_doing << _pg->myrank() << " Create the graph and stencil" << "\n";
  HYPRE_SStructGraphCreate(_pg->getComm(), _grid, &_graph);
  _exists[SStructGraph] = SStructCreated;
  
  // For ParCSR-requiring solvers like AMG
  if (_requiresPar) {
    HYPRE_SStructGraphSetObjectType(_graph, HYPRE_PARCSR);
  }

  //  Add stencil-based equations to the interior of the graph.
  for (int level = 0; level < numLevels; level++) {
    HYPRE_SStructGraphSetStencil(_graph, level, CC_VAR, _stencil);
  }
  // Add to graph the unstructured part of the stencil connecting the
  // coarse and fine levels at every C/F boundary   
  // Note: You need to do this "looking up" and "looking down'      
  for (int p = 0; p < _patches->size(); p++) {
    const Patch* patch = _patches->get(p);
    HyprePatch_CC hpatch(_patches->get(p),matl);
    int level = hpatch.getLevel();
    
    // Looking down
    if ((level > 0) && (patch->hasCoarseFaces())) {
      hpatch.makeGraphConnections(_graph,DoingFineToCoarse);
    } 
    // Looking up
    if (level < numLevels-1) {
      hpatch.makeGraphConnections(_graph,DoingCoarseToFine);
    }
  } 
  cout_doing << Parallel::getMPIRank()<< " Doing Assemble graph \t\t\tPatches"<< *_patches << endl;  
  HYPRE_SStructGraphAssemble(_graph);
  _exists[SStructGraph] = SStructAssembled;

  //==================================================================
  // Set up matrix _HA
  //==================================================================
  HYPRE_SStructMatrixCreate(_pg->getComm(), _graph, &_HA);
  _exists[SStructA] = SStructCreated;
  // If specified by input parameter, declare the structured and
  // unstructured part of the matrix to be symmetric.
  
  for (int level = 0; level < numLevels; level++) {
    HYPRE_SStructMatrixSetSymmetric(_HA, level,
                                    CC_VAR, CC_VAR,
                                    _params->symmetric);
  }
  HYPRE_SStructMatrixSetNSSymmetric(_HA, _params->symmetric);

  // For solvers that require ParCSR format
  if (_requiresPar) {
    HYPRE_SStructMatrixSetObjectType(_HA, HYPRE_PARCSR);
  }
  HYPRE_SStructMatrixInitialize(_HA);
  _exists[SStructA] = SStructInitialized;

  //__________________________________
  // added the stencil entries to the interior cells
  for (int p = 0 ; p < _patches->size(); p++) {
    const Patch* patch = _patches->get(p);
    HyprePatch_CC hpatch(patch,matl); 
    hpatch.makeInteriorEquations(_HA, _A_dw, _A_label,
                                 _stencilSize, _params->symmetric);
  }
  
  for (int l = 0; l < numLevels; l++) {
    if (useBogusLevelData[l]) {
      HyprePatch_CC hpatch(l, matl);
      hpatch.makeInteriorEquations(_HA, _A_dw, _A_label, 
                                   _stencilSize, _params->symmetric);
    }
  }
  //__________________________________
  // added the unstructured entries at the C/F interfaces
  for (int p = 0; p < _patches->size(); p++) {
    const Patch* patch = _patches->get(p);
    HyprePatch_CC hpatch(_patches->get(p),matl); 
    int level = hpatch.getLevel();
    
    // Looking Down
    if ((level > 0) && (patch->hasCoarseFaces())) {
      hpatch.makeConnections(_HA, _A_dw, _A_label,
                             _stencilSize, DoingFineToCoarse);
    }
    // Looking up
    if (level < numLevels-1) {
      hpatch.makeConnections(_HA, _A_dw, _A_label,
                             _stencilSize, DoingCoarseToFine);
    }
  } 
  HYPRE_SStructMatrixAssemble(_HA);
  _exists[SStructA] = SStructAssembled;

  //==================================================================
  //  Create the rhs
  //==================================================================
  cout_doing << _pg->myrank() << " Doing setup RHS vector _HB" << "\n";
  HYPRE_SStructVectorCreate(_pg->getComm(), _grid, &_HB);
  _exists[SStructB] = SStructCreated;
  
  if (_requiresPar) {
    HYPRE_SStructVectorSetObjectType(_HB, HYPRE_PARCSR);
  }
  HYPRE_SStructVectorInitialize(_HB);
  _exists[SStructB] = SStructInitialized;

  // Set RHS vector entries
  for (int p = 0 ; p < _patches->size(); p++) {
    const Patch* patch = _patches->get(p);
    HyprePatch_CC hpatch(patch,matl); 
    hpatch.makeInteriorVector(_HB, _b_dw, _B_label);
  } 
  
  for (int l = 0; l < numLevels; l++) {
    if (useBogusLevelData[l]) {
      HyprePatch_CC hpatch(l, matl);
      hpatch.makeInteriorVectorZero(_HB, _b_dw, _B_label);
    }
  }

  HYPRE_SStructVectorAssemble(_HB);
  _exists[SStructB] = SStructAssembled;
  //==================================================================
  //  Create the solution
  //==================================================================
  cout_doing << _pg->myrank() << " Doing setup solution vector _HX" << "\n";
  HYPRE_SStructVectorCreate(_pg->getComm(), _grid, &_HX);
  _exists[SStructX] = SStructCreated;
  
  if (_requiresPar) {
    HYPRE_SStructVectorSetObjectType(_HX, HYPRE_PARCSR);
  }
  HYPRE_SStructVectorInitialize(_HX);
  _exists[SStructX] = SStructInitialized;
  
  if (_guess_label) {
    for (int p = 0 ; p < _patches->size(); p++) {
      const Patch* patch = _patches->get(p);
      HyprePatch_CC hpatch(patch,matl); 
      hpatch.makeInteriorVector(_HX, _guess_dw, _guess_label);
    } 
    
    for (int l = 0; l < numLevels; l++) {
      if (useBogusLevelData[l]) {
        HyprePatch_CC hpatch(l, matl);
        hpatch.makeInteriorVectorZero(_HX, _guess_dw, _guess_label);
      }
    }
  } else {
#if 0
    // If guess is not provided by ICE, use zero as initial guess
    cout_doing << _pg->myrank() << " Default initial guess: zero" << "\n";
    for (int p = 0 ; p < _patches->size(); p++) {
      // Read Uintah patch info into our data structure, set Uintah pointers
      const Patch* patch = _patches->get(p);
      HyprePatch_CC hpatch(patch,matl); // Read Uintah patch data
      hpatch.makeInteriorVectorZero(_HX, _guess_dw, _guess_label);
    } // end for p (patches)
    for (int l = 0; l < numLevels; l++) {
      if (useBogusLevelData[l]) {
        HyprePatch_CC hpatch(l, matl);
        hpatch.makeInteriorVectorZero(_HX, _guess_dw, _guess_label);
      }
    }
#endif
  }

  HYPRE_SStructVectorAssemble(_HX);
  _exists[SStructX] = SStructAssembled;

  // For solvers that require ParCSR format
  if (_requiresPar) {
    cout_doing << _pg->myrank() << " Making ParCSR objects from SStruct objects" << "\n";
    HYPRE_SStructMatrixGetObject(_HA, (void **) &_HA_Par);
    HYPRE_SStructVectorGetObject(_HB, (void **) &_HB_Par);
    HYPRE_SStructVectorGetObject(_HX, (void **) &_HX_Par);
  }
  cout_doing << Parallel::getMPIRank() << " HypreDriverSStruct::makeLinearSystem_CC() END" << "\n";
} // end HypreDriverSStruct::makeLinearSystem_CC()


//_____________________________________________________________________*/
void
HypreDriverSStruct::getSolution_CC(const int matl)
{
  for (int p = 0 ; p < _patches->size(); p++) {
    HyprePatch_CC hpatch(_patches->get(p),matl); // Read Uintah patch data
    hpatch.getSolution(_HX,_new_dw,_X_label,_modifies_x);
  } 
} 

//___________________________________________________________________
// Add this patch to the Hypre grid
//___________________________________________________________________
void
HypreDriverSStruct::HyprePatch_CC::addToGrid(HYPRE_SStructGrid& grid,
                                             HYPRE_SStructVariable* vars)

{
  cout_doing << Parallel::getMPIRank() << " Adding patch " << (_patch?_patch->getID():-1)
       << " from "<< _low << " to " << _high
       << " Level " << _level << "\n";

  HYPRE_SStructGridSetExtents(grid, _level,
                              _low.get_pointer(),
                              _high.get_pointer());
  HYPRE_SStructGridSetVariables(grid, _level, CC_NUM_VARS, vars);
}
//___________________________________________________________________
// HypreDriverSStruct::HyprePatch_CC::makeGraphConnections~
// Add the connections at C/F interfaces of this patch to the HYPRE
// Graph.   You must do for the graph "looking up" from the coarse patch
// and "looking down" from the finePatch.
//___________________________________________________________________
void
HypreDriverSStruct::HyprePatch_CC::makeGraphConnections(HYPRE_SStructGraph& graph,
                                                   const CoarseFineViewpoint& viewpoint)

{
  int mpiRank = Parallel::getMPIRank();
  cout_doing << mpiRank << " Doing makeGraphConnections \t\t\t\tL-"
                  << _level << " Patch " << _patch->getID()
                  << " viewpoint " << viewpoint << endl;

  //__________________________________
  // viewpoint LOGIC
  int fineIndex, coarseIndex;
  Level::selectType finePatches;
  Level::selectType coarsePatches;
  const Level* fineLevel;
  const Level* coarseLevel;
  fineLevel=0;
  coarseLevel=0;
  
  if(viewpoint == DoingFineToCoarse){  // looking down
    const Patch* finePatch   = _patch;
    fineLevel   = finePatch->getLevel();
    coarseLevel = fineLevel->getCoarserLevel().get_rep();
    
    finePatches.push_back(finePatch);
    finePatch->getCoarseLevelPatches(coarsePatches);
  }
  if(viewpoint == DoingCoarseToFine){   // looking up
    const Patch* coarsePatch = _patch;
    coarseLevel = coarsePatch->getLevel();
    fineLevel   = coarseLevel->getFinerLevel().get_rep();
    
    coarsePatches.push_back(coarsePatch);
    coarsePatch->getFineLevelPatches(finePatches);
  }
  
  coarseIndex  = coarseLevel->getIndex();
  fineIndex  = fineLevel->getIndex();

  const IntVector& refRat = fineLevel->getRefinementRatio();
  
  //At the CFI compute the fine/coarse level indices and pass them to hypre
  for(int i = 0; i < coarsePatches.size(); i++){  
    const Patch* coarsePatch = coarsePatches[i];
    
    for(int i = 0; i < finePatches.size(); i++){  
      const Patch* finePatch = finePatches[i];

      vector<Patch::FaceType> cf;
      finePatch->getCoarseFaces(cf);
      vector<Patch::FaceType>::const_iterator iter; 
      for (iter  = cf.begin(); iter != cf.end(); ++iter) {

        Patch::FaceType face = *iter;                   
        IntVector offset = finePatch->faceDirection(face);

        bool isRight_CP_FP_pair = false;
        CellIterator f_iter(IntVector(-8,-8,-8),IntVector(-9,-9,-9));
        fineLevel_CFI_Iterator(face, coarsePatch, finePatch, f_iter, isRight_CP_FP_pair);
        if(isRight_CP_FP_pair){
#ifdef DEBUG       // spew
          cout << mpiRank << "-----------------Face " << finePatch->getFaceName(face) 
               << " iter " << f_iter.begin() << " " << f_iter.end() 
               << " offset " << offset
               << " finePatch ID " << finePatch->getID() 
               << " f_level " << fineIndex 
               << " c_level " << coarseIndex << endl;
#endif

          for(; !f_iter.done(); f_iter++) {
            IntVector fineCell = *f_iter;                        
            IntVector coarseCell = (fineCell + offset) / refRat;

            if(viewpoint == DoingFineToCoarse){

              //cout <<mpiRank<<" looking Down: fineCell " << fineCell 
              //   << " -> coarseCell " << coarseCell;

              HYPRE_SStructGraphAddEntries(graph,
                                           fineIndex, fineCell.get_pointer(),
                                           CC_VAR,
                                           coarseIndex, coarseCell.get_pointer(),
                                           CC_VAR);
              //cout << " done " << endl;

            }
            if(viewpoint == DoingCoarseToFine){
              //cout <<mpiRank<<" looking Up: fineCell " << fineCell 
              //     << " <- coarseCell " << coarseCell;

              HYPRE_SStructGraphAddEntries(graph,
                                           coarseIndex, coarseCell.get_pointer(),
                                           CC_VAR,
                                           fineIndex, fineCell.get_pointer(),
                                           CC_VAR);
              //cout << " done " << endl;

            }
          }
        }
      } // CFI
    }  // finePatches
  } // coarsePatches
} 
//___________________________________________________________________
// HypreDriverSStruct::HyprePatch_CC::makeInteriorEquations~
// Add the connections at C/F interfaces of this patch to the HYPRE
// Graph. If viewpoint == DoingFineToCoarse, we add the fine-to-coarse
// connections. If viewpoint == DoingCoarseToFine, we add the
// coarse-to-fine-connections that are read from the connection list
// prepared for this patch by ICE.
//___________________________________________________________________
void
HypreDriverSStruct::HyprePatch_CC::makeInteriorEquations(HYPRE_SStructMatrix& HA,
                                                         DataWarehouse* A_dw,
                                                         const VarLabel* A_label,
                                                         const int stencilSize,
                                                         const bool symmetric /* = false */)

{
  cout_doing << Parallel::getMPIRank() << " doing makeInteriorEquations \t\t\t\tL-" 
             << _level<< " Patch " << (_patch?_patch->getID():-1) << endl;

  CCTypes::matrix_type A;
  if (_patch) {
    A_dw->get(A, A_label, _matl, _patch, Ghost::None, 0);
  }
  else {
    // should be a 1-cell object
    ASSERT(_low == _high);
    CCVariable<Stencil7>& Amod = A.castOffConst();
    Amod.rewindow(_low, _high +IntVector(1,1,1));
    Amod[_low].w = Amod[_low].e = Amod[_low].s = Amod[_low].n = Amod[_low].b = Amod[_low].t = 0;
    Amod[_low].p = 1;
  }
  
  if (symmetric) {
    double* values = scinew double[(_high.x()-_low.x()+1)*stencilSize];
    int stencil_indices[] = {0,1,2,3};
    for(int z = _low.z(); z <= _high.z(); z++) {
      for(int y = _low.y(); y <= _high.y(); y++) {
        const Stencil7* AA = &A[IntVector(_low.x(), y, z)];
        double* p = values;
        for (int x = _low.x(); x <= _high.x(); x++) {
          // Keep the ordering as in stencil offsets:
          // w s b p
          *p++ = AA->w;
          *p++ = AA->s;
          *p++ = AA->b;
          *p++ = AA->p;
          AA++;
        }
        IntVector chunkLow(_low.x(), y, z);
        IntVector chunkHigh(_high.x(), y, z);
        HYPRE_SStructMatrixSetBoxValues(HA, _level,
                                        chunkLow.get_pointer(),
                                        chunkHigh.get_pointer(), CC_VAR,
                                        stencilSize, stencil_indices, values);
      }
    }
    delete[] values;
  } else { // now symmetric = false
    double* values = scinew double[(_high.x()-_low.x()+1)*stencilSize];
    int stencil_indices[] = {0,1,2,3,4,5,6};
    for(int z = _low.z(); z <= _high.z(); z++) {
      for(int y = _low.y(); y <= _high.y(); y++) {
        const Stencil7* AA = &A[IntVector(_low.x(), y, z)];
        double* p = values;
        for (int x = _low.x(); x <= _high.x(); x++) {
          // Keep the ordering as in stencil offsets:
          // w e s n b t p
          *p++ = AA->w;
          *p++ = AA->e;
          *p++ = AA->s;
          *p++ = AA->n;
          *p++ = AA->b;
          *p++ = AA->t;
          *p++ = AA->p;
          AA++;
        }
        IntVector chunkLow(_low.x(), y, z);
        IntVector chunkHigh(_high.x(), y, z);
        HYPRE_SStructMatrixSetBoxValues(HA, _level,
                                        chunkLow.get_pointer(),
                                        chunkHigh.get_pointer(), CC_VAR,
                                        stencilSize, stencil_indices, values);
      }
    }
    delete[] values;
  } 
}

//___________________________________________________________________
//makeInteriorVector.
//___________________________________________________________________
void
HypreDriverSStruct::HyprePatch_CC::makeInteriorVector(HYPRE_SStructVector& HV,
                                                      DataWarehouse* V_dw,
                                                      const VarLabel* V_label)
{
  CCTypes::const_type V;
  V_dw->get(V, V_label, _matl, _patch, Ghost::None, 0);
  
  for(int z = _low.z(); z <= _high.z(); z++) {
    for(int y = _low.y(); y <= _high.y(); y++) {
      const double* values = &V[IntVector(_low.x(), y, z)];
      IntVector chunkLow(_low.x(), y, z);
      IntVector chunkHigh(_high.x(), y, z);
      HYPRE_SStructVectorSetBoxValues(HV, _level,
                                      chunkLow.get_pointer(),
                                      chunkHigh.get_pointer(), CC_VAR,
                                      const_cast<double*>(values));
    }
  }
} 

//___________________________________________________________________
// makeInteriorVectorZero
//___________________________________________________________________
void
HypreDriverSStruct::HyprePatch_CC::makeInteriorVectorZero(HYPRE_SStructVector& HV,
                                                          DataWarehouse* V_dw,
                                                          const VarLabel* V_label)
{
  // Make a vector of zeros
  CCVariable<double> V;
  V.rewindow(_low, _high+IntVector(1,1,1));
  V.initialize(0.0);
  
  for(int z = _low.z(); z <= _high.z(); z++) {
    for(int y = _low.y(); y <= _high.y(); y++) {
      const double* values = &V[IntVector(_low.x(), y, z)];
      IntVector chunkLow(_low.x(), y, z);
      IntVector chunkHigh(_high.x(), y, z);
      HYPRE_SStructVectorSetBoxValues(HV, _level,
                                      chunkLow.get_pointer(),
                                      chunkHigh.get_pointer(), CC_VAR,
                                      const_cast<double*>(values));
    }
  }
} 


//___________________________________________________________________
// HypreDriverSStruct::HyprePatch_CC::makeConnections~
// Add the connections at C/F interfaces of this patch.
//___________________________________________________________________
void
HypreDriverSStruct::HyprePatch_CC::makeConnections(HYPRE_SStructMatrix& HA,
                                                   DataWarehouse* A_dw,
                                                   const VarLabel* A_label,
                                                   const int stencilSize,
                                                   const CoarseFineViewpoint& viewpoint)

{
  int mpiRank = Parallel::getMPIRank();
  cout_doing << mpiRank << " Doing makeConnections \t\t\t\tL-"
                        << _level << " Patch " << _patch->getID()
                        << " viewpoint " << viewpoint << endl;

  //__________________________________
  // viewpoint LOGIC
  int fineIndex, coarseIndex;
  Level::selectType finePatches;
  Level::selectType coarsePatches;
  const Level* fineLevel = NULL;
  const Level* coarseLevel = NULL;
  const double ZERO = 0.0;
  //__________________________________
  // looking down
  if(viewpoint == DoingFineToCoarse){  
    const Patch* finePatch   = _patch;
    fineLevel   = finePatch->getLevel();
    coarseLevel = fineLevel->getCoarserLevel().get_rep();
    
    finePatches.push_back(finePatch);
    finePatch->getCoarseLevelPatches(coarsePatches);

  }
  //__________________________________
  // looking up
  if(viewpoint == DoingCoarseToFine){   
    const Patch* coarsePatch = _patch;
    coarseLevel = coarsePatch->getLevel();
    fineLevel   = coarseLevel->getFinerLevel().get_rep();
    
    coarsePatches.push_back(coarsePatch);
    coarsePatch->getFineLevelPatches(finePatches);
  }
  
  coarseIndex  = coarseLevel->getIndex();
  fineIndex  = fineLevel->getIndex();

  const IntVector& refRat = fineLevel->getRefinementRatio();
  
  //At the CFI compute the fine/coarse level indices and pass them to hypre
  for(int i = 0; i < coarsePatches.size(); i++){  
    const Patch* coarsePatch = coarsePatches[i];
    
    for(int i = 0; i < finePatches.size(); i++){  
      const Patch* finePatch = finePatches[i];
      
      //__________________________________
      // get the fine level data
      CCTypes::matrix_type A_fine;
      CCVariable<int> counter_fine, counter_coarse;
      
      if(viewpoint == DoingFineToCoarse){    
        A_dw->get(A_fine, A_label, _matl, finePatch, Ghost::None, 0);
          
        A_dw->allocateTemporary(counter_fine, finePatch);
        counter_fine.initialize(stencilSize);  
      }
      if(viewpoint == DoingCoarseToFine){
        IntVector cl, ch, fl, fh;
        int nGhostCells = 1;
        IntVector bl(0,0,0);  // boundary layer cells
        bool returnExclusiveRange=true;
        getCoarseLevelRange(finePatch, coarseLevel, cl, ch, fl, fh, bl, 
                            nGhostCells, returnExclusiveRange);
                            
        A_dw->getRegion(A_fine, A_label, _matl, fineLevel, fl, fh);
        
        counter_coarse.allocate(cl, ch);
        counter_coarse.initialize(stencilSize );
      } 

      vector<Patch::FaceType> cf;
      finePatch->getCoarseFaces(cf);
      vector<Patch::FaceType>::const_iterator iter; 
      for (iter  = cf.begin(); iter != cf.end(); ++iter) {

        Patch::FaceType face = *iter;                   
        IntVector offset = finePatch->faceDirection(face);
        int opposite = face - int(patchFaceSide(face));    
        int stencilIndex_fine[1]   = {face};
        int stencilIndex_coarse[1] = {opposite};

        bool isRight_CP_FP_pair = false;
        CellIterator f_iter(IntVector(-8,-8,-8),IntVector(-9,-9,-9));
        fineLevel_CFI_Iterator(face, coarsePatch, finePatch, f_iter, isRight_CP_FP_pair);
        
        if(isRight_CP_FP_pair){
#ifdef DEBUG           
          cout << mpiRank << "-----------------Face " << finePatch->getFaceName(face) 
               << " iter " << f_iter.begin() << " " << f_iter.end() 
               << " offset " << offset
               << " finePatch ID " << finePatch->getID() 
               << " f_level " << fineIndex 
               << " c_level " << coarseIndex << endl;
#endif

          for(; !f_iter.done(); f_iter++) {
            IntVector fineCell = *f_iter;                        
            IntVector coarseCell = (fineCell + offset) / refRat;

            if(viewpoint == DoingFineToCoarse){   // ----------------------------


              //  Update the entries on the fine level
              int graphIndex_fine[1] = {counter_fine[fineCell]};
              const double* graphValue = &A_fine[fineCell][face];

              // add the unstructured entry to the matrix
              HYPRE_SStructMatrixSetValues(HA, fineIndex,
                                           fineCell.get_pointer(),
                                           CC_VAR, 1, graphIndex_fine,
                                           const_cast<double*>(graphValue));

              // Wipe out the original structured entry 
              const double* stencilValue = &ZERO;
               HYPRE_SStructMatrixSetValues(HA, fineIndex,
                                           fineCell.get_pointer(),
                                           CC_VAR, 1, stencilIndex_fine,
                                           const_cast<double*>(stencilValue));

              counter_fine[fineCell]++;
        #ifdef DEBUG
              cout << " looking Down: finePatch "<< fineCell
                   << " f_index " << graphIndex_fine[0]
                   << " s_index " << stencilIndex_fine[0]
                   << " value " << graphValue[0] 
                   << " \t| Coarse " << coarseCell
                   << " s_index " << stencilIndex_coarse[0]<<endl;  
        #endif
            }
            if(viewpoint == DoingCoarseToFine){     // ----------------------------


              IntVector coarseCell = (fineCell + offset) / refRat;
              int graphIndex_coarse[1] = {counter_coarse[coarseCell]};
              const double* graphValue = &A_fine[fineCell][face];

              HYPRE_SStructMatrixSetValues(HA, coarseIndex,
                                           coarseCell.get_pointer(),
                                           CC_VAR, 1, graphIndex_coarse,
                                           const_cast<double*>(graphValue));

              // Wipe out the original coarse-coarse structured connection
              if (counter_coarse[coarseCell] == stencilSize) {
                const double* stencilValue = &ZERO;
                HYPRE_SStructMatrixSetValues(HA, coarseIndex,
                                             coarseCell.get_pointer(),
                                             CC_VAR, 1, stencilIndex_coarse, 
                                             const_cast<double*>(stencilValue));
              } 
              counter_coarse[coarseCell]++;
        #ifdef DEBUG
              cout << " looking Up: finePatch "<< fineCell
                   << " s_index " << stencilIndex_fine[0]
                   << " value " << graphValue[0] 
                   << " \t| Coarse " << coarseCell
                   << " c_index " << graphIndex_coarse[0]
                   << " s_index " << stencilIndex_coarse[0]<<endl;  
        #endif
            }
          }
        }
      } // CFI
    }  // finePatches
  } // coarsePatches
}

//___________________________________________________________________
// getSolution(): move Hypre solution into Uintah datastructure
//___________________________________________________________________
void
HypreDriverSStruct::HyprePatch_CC::getSolution(HYPRE_SStructVector& HX,
                                               DataWarehouse* new_dw,
                                               const VarLabel* X_label,
                                               const bool modifies_x)
{
  typedef CCTypes::sol_type sol_type;
  sol_type Xnew;
  if (modifies_x) {
    new_dw->getModifiable(Xnew, X_label, _matl, _patch);
  } else {
    new_dw->allocateAndPut(Xnew, X_label, _matl, _patch);
  }

  for(int z = _low.z(); z <= _high.z(); z++) {
    for(int y = _low.y(); y <= _high.y(); y++) {
      const double* values = &Xnew[IntVector(_low.x(), y, z)];
      IntVector chunkLow(_low.x(), y, z);
      IntVector chunkHigh(_high.x(), y, z);
      HYPRE_SStructVectorGetBoxValues(HX, _level,
                                      chunkLow.get_pointer(),
                                      chunkHigh.get_pointer(),
                                      CC_VAR, const_cast<double*>(values));
    }
  }
}

//___________________________________________________________________
// Utilities
//___________________________________________________________________
void printLine(const string& s, const unsigned int len)
{
  for (unsigned int i = 0; i < len; i++) {
    cout << s;
  }
  cout << "\n";
}


namespace Uintah {

  std::ostream&
  operator << (std::ostream& os,
               const HypreDriverSStruct::CoarseFineViewpoint& v)
  {
    if      (v == HypreDriverSStruct::DoingCoarseToFine) os << "CoarseToFine";
    else if (v == HypreDriverSStruct::DoingFineToCoarse) os << "FineToCoarse";
    else os << "CoarseFineViewpoint WRONG!!!";
    return os;
  }

  std::ostream& operator<< (std::ostream& os,
                            const HypreDriverSStruct::HyprePatch& p)
    // Write our patch structure to the stream os.
  {
    os << *(p.getPatch());
    return os;
  }


} // end namespace Uintah

