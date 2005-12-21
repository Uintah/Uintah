/*--------------------------------------------------------------------------
 * File: HypreDriverSStruct.cc
 *
 * Implementation of a wrapper of a Hypre solvers working with the Hypre
 * SStruct system interface.
 *--------------------------------------------------------------------------*/

#include <sci_defs/hypre_defs.h>

#include <Packages/Uintah/CCA/Components/Solvers/HypreDriverSStruct.h>
#include <Packages/Uintah/CCA/Components/Solvers/MatrixUtil.h>

#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Packages/Uintah/Core/Grid/Variables/Stencil7.h>
#include <Packages/Uintah/Core/Grid/Variables/VarTypes.h>
#include <Packages/Uintah/Core/Grid/Variables/AMRInterpolate.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/Core/Parallel/Parallel.h>

#include <Core/Geometry/IntVector.h>
#include <Core/Math/MiscMath.h>
#include <Core/Util/DebugStream.h>

#include <iomanip>
#include <iostream>

using namespace Uintah;
using namespace std;

//__________________________________
//  To turn on normal output
//  setenv SCI_DEBUG "HYPRE_DOING_COUT:+"

static DebugStream cout_doing("HYPRE_DOING_COUT", false);
static DebugStream cout_dbg("HYPRE_DBG", false);

//#####################################################################
// class HypreDriver implementation common to all variable types
//#####################################################################

HypreDriverSStruct::HyprePatch::HyprePatch(const Patch* patch,
                                           const int matl) :
  //___________________________________________________________________
  // HypreDriverSStruct::HyprePatch constructor from Uintah patch
  //___________________________________________________________________
  _patch(patch), _matl(matl),
  _level(patch->getLevel()->getIndex()),
  _low(patch->getInteriorCellLowIndex()),
  _high(patch->getInteriorCellHighIndex()-IntVector(1,1,1))
  // Note: we need to subtract (1,1,1) from high because our loops are
  // cell = low; cell <= high.
{
}

HypreDriverSStruct::HyprePatch::HyprePatch(const int level,
                                           const int matl) :
  //___________________________________________________________________
  // HypreDriverSStruct::HyprePatch bogus patch constructor
  //   for when there are no patches on a level for a processor
  //   this is a one celled "patch"
  //___________________________________________________________________
  _patch(0), _matl(matl),
  _level(level),
  _low(IntVector(-9*(level+1),-9,-9)),
  _high(IntVector(-9*(level+1),-9,-9))
{
}

HypreDriverSStruct::HyprePatch::~HyprePatch(void)
{}

HypreDriverSStruct::~HypreDriverSStruct(void)
  //___________________________________________________________________
  // HypreDriverSStruct destructor
  //___________________________________________________________________
{
  cout_doing << "HypreDriverSStruct destructor BEGIN" << "\n";
  printDataStatus();
  // Destroy matrix, RHS, solution objects
  if (_exists[SStructA] >= SStructAssembled) {
    HYPRE_SStructMatrixDestroy(_HA);
    _exists[SStructA] = SStructDestroyed;
  }
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
  if (_vars) {
    delete _vars;
  }
  if (_exists[SStructGrid] >= SStructAssembled) {
    HYPRE_SStructGridDestroy(_grid);
    _exists[SStructGrid] = SStructDestroyed;
  }
}
//______________________________________________________________________
void
HypreDriverSStruct::printMatrix(const string& fileName /* =  "output" */)
{
  if (!_params->printSystem) return;
  HYPRE_SStructMatrixPrint((fileName + ".sstruct").c_str(), _HA, 0);
  if (_requiresPar) {
    HYPRE_ParCSRMatrixPrint(_HA_Par, (fileName + ".par").c_str());
    // Print CSR matrix in IJ format, base 1 for rows and cols
#if HAVE_HYPRE_1_9
    HYPRE_ParCSRMatrixPrintIJ(_HA_Par, 1, 1, (fileName + ".ij").c_str());
#else
    cerr << "Warning: this Hypre version does not support printing the "
         << "matrix in IJ format to a file, skipping this printout" << "\n";
#endif // #if HAVE_HYPRE_1_9
  }
}
//______________________________________________________________________
void
HypreDriverSStruct::printRHS(const string& fileName /* =  "output_b" */)
{
  if (!_params->printSystem) return;
  HYPRE_SStructVectorPrint(fileName.c_str(), _HB, 0);
  if (_requiresPar) {
    HYPRE_ParVectorPrint(_HB_Par, (fileName + ".par").c_str());
  }
}
//______________________________________________________________________
void
HypreDriverSStruct::printSolution(const string& fileName /* =  "output_x" */)
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

void
HypreDriverSStruct::printDataStatus(void)
{
  cout_dbg << "Hypre SStruct interface data status:" << "\n";
  for (unsigned i = 0; i < _exists.size(); i++) {
    cout_dbg << "_exists[" << i << "] = " << _exists[i] << "\n";
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
  cout_doing << "HypreDriverSStruct::makeLinearSystem_CC() BEGIN" << "\n";
  ASSERTEQ(sizeof(Stencil7), 7*sizeof(double));
  
  //__________________________________
  // Set up the grid
  cout_dbg << _pg->myrank() << " Setting up the SStruct grid" << "\n";
  // Create an empty grid in 3 dimensions with # parts = numLevels.
  const int numDims = 3;
  const int numLevels = _level->getGrid()->numLevels();
  
  HYPRE_SStructGridCreate(_pg->getComm(), numDims, numLevels, &_grid);
  cout_dbg << _pg->myrank() << " Constructed empty grid, numDims " << numDims
       << " numParts " << numLevels << "\n";
  _exists[SStructGrid] = SStructCreated;
  _vars = new HYPRE_SStructVariable[CC_NUM_VARS];
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
  cout_doing << _pg->myrank() << " Setting up the SStruct graph" << "\n";
  HYPRE_SStructGraphCreate(_pg->getComm(), _grid, &_graph);
  _exists[SStructGraph] = SStructCreated;
  
  // For ParCSR-requiring solvers like AMG
  if (_requiresPar) {
#if HAVE_HYPRE_1_9
    HYPRE_SStructGraphSetObjectType(_graph, HYPRE_PARCSR);
#else
    ostringstream msg;
    msg << "Hypre version does not support solvers that require "
        << "conversion from SStruct to ParCSR" << "\n";
    throw InternalError(msg.str(),__FILE__, __LINE__);
#endif 
  }

  //  Add stencil-based equations to the interior of the graph.
  for (int level = 0; level < numLevels; level++) {
    HYPRE_SStructGraphSetStencil(_graph, level, CC_VAR, _stencil);
  }

  //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
  // Add to graph the unstructured part of the stencil connecting the
  // coarse and fine level at every C/F boundary (F->C connections at
  // this patch's outer boundaries, and C->F connections at all
  // applicable C/F boundaries of next-finer-level patches that lie
  // above this patch.
  //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

  for (int p = 0; p < _patches->size(); p++) {

    const Patch* patch = _patches->get(p);
    HyprePatch_CC hpatch(_patches->get(p),matl);
    int level = hpatch.getLevel();
    
    //__________________________________
    //  fine level: coarse fine interfaces
    if ((level > 0) && (patch->hasCoarseFineInterfaceFace())) {
      hpatch.makeConnections(_graph,DoingFineToCoarse);
    } 

    if (level < numLevels-1) {
      // If not at finest level, examine the connection list that
      // impAMRICE.cc provides us and add the coarse-to-fine
      // connections to the Hypre graph. This list should cover all
      // the C/F interfaces of all next-finer level patches inscribed
      // in this patch.
      printLine("=",50);
      cout_dbg << _pg->myrank() << " Building coarse-to-fine connections" << "\n";
      printLine("=",50);
      hpatch.makeConnections(_graph,DoingCoarseToFine);
    } 
  } 
  
  HYPRE_SStructGraphAssemble(_graph);
  _exists[SStructGraph] = SStructAssembled;

  //==================================================================
  // Set up matrix _HA
  //==================================================================
  HYPRE_SStructMatrixCreate(_pg->getComm(), _graph, &_HA);
  _exists[SStructA] = SStructCreated;
  // If specified by input parameter, declare the structured and
  // unstructured part of the matrix to be symmetric.
  
#if HAVE_HYPRE_1_9
  for (int level = 0; level < numLevels; level++) {
    HYPRE_SStructMatrixSetSymmetric(_HA, level,
                                    CC_VAR, CC_VAR,
                                    _params->symmetric);
  }
  HYPRE_SStructMatrixSetNSSymmetric(_HA, _params->symmetric);
#else
  cerr << "Warning: Hypre version does not correctly support "
       << "symmetric matrices; proceeding without doing anything "
       << "at this point." << "\n";
#endif

  // For solvers that require ParCSR format
  if (_requiresPar) {
    HYPRE_SStructMatrixSetObjectType(_HA, HYPRE_PARCSR);
  }
  HYPRE_SStructMatrixInitialize(_HA);
  _exists[SStructA] = SStructInitialized;

  //__________________________________
  // added the stencil entries to the interior cells
  cout_doing << _pg->myrank() << " Matrix structured (interior) entries" << "\n";
  
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
  cout_doing << _pg->myrank() << " Matrix unstructured (C/F) entries" << "\n";

  for (int p = 0; p < _patches->size(); p++) {
    const Patch* patch = _patches->get(p);
    HyprePatch_CC hpatch(_patches->get(p),matl); 
    int level = hpatch.getLevel();
    
    if ((level > 0) && (patch->hasCoarseFineInterfaceFace())) {
      // If not at coarsest level, add fine-to-coarse connections at all
      // C/F interface faces.
      hpatch.makeConnections(_HA, _A_dw, _A_label,
                             _stencilSize, DoingFineToCoarse);
    } 
    
    if (level < numLevels-1) {
      // If not at finest level, examine the connection list that
      // impAMRICE.cc provides us and add the coarse-to-fine
      // connections to the Hypre graph. This list should cover all
      // the C/F interfaces of all next-finer level patches inscribed
      // in this patch.
      hpatch.makeConnections(_HA, _A_dw, _A_label,
                             _stencilSize, DoingCoarseToFine);
    } 
  } 

  HYPRE_SStructMatrixAssemble(_HA);
  _exists[SStructA] = SStructAssembled;

  //==================================================================
  //  Create the rhs
  //==================================================================
  cout_dbg << _pg->myrank() << " Setting up the SStruct RHS vector _HB" << "\n";
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
  cout_dbg << _pg->myrank() << " Setting up the SStruct solution vector _HX" << "\n";
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
    cout_dbg << _pg->myrank() << " Default initial guess: zero" << "\n";
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
    cout_dbg << _pg->myrank() << " Making ParCSR objects from SStruct objects" << "\n";
    HYPRE_SStructMatrixGetObject(_HA, (void **) &_HA_Par);
    HYPRE_SStructVectorGetObject(_HB, (void **) &_HB_Par);
    HYPRE_SStructVectorGetObject(_HX, (void **) &_HX_Par);
  }
  cout_doing << "HypreDriverSStruct::makeLinearSystem_CC() END" << "\n";
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
  cout_dbg << Parallel::getMPIRank() << " Adding patch " << (_patch?_patch->getID():-1)
       << " from "<< _low << " to " << _high
       << " Level " << _level << "\n";
  HYPRE_SStructGridSetExtents(grid, _level,
                              _low.get_pointer(),
                              _high.get_pointer());
  HYPRE_SStructGridSetVariables(grid, _level, CC_NUM_VARS, vars);
}

//___________________________________________________________________
// HypreDriverSStruct::HyprePatch_CC::makeConnections~
// Add the connections at C/F interfaces of this patch to the HYPRE
// Graph. If viewpoint == DoingFineToCoarse, we add the fine-to-coarse
// connections. If viewpoint == DoingCoarseToFine, we add the
// coarse-to-fine-connections that are read from the connection list
// prepared for this patch by ICE.
//___________________________________________________________________
void
HypreDriverSStruct::HyprePatch_CC::makeConnections(HYPRE_SStructGraph& graph,
                                                   const CoarseFineViewpoint& viewpoint)

{
  cout_doing << "Doing makeConnections graph for Patch "
             << _patch->getID() << " Level " << _level << "\n";
       
  const GridP grid = _patch->getLevel()->getGrid();
  
    // Add fine-to-coarse connections to graph
  if (viewpoint == DoingFineToCoarse) {
    cout_doing << "Adding fine-to-coarse connections to graph" << "\n";
    
    const int fineLevel   = _level;
    const int coarseLevel = _level-1;
    const IntVector& refRat = grid->getLevel(fineLevel)->getRefinementRatio();
    vector<Patch::FaceType>::const_iterator iter;  

    for (iter = _patch->getCoarseFineInterfaceFaces()->begin(); 
         iter != _patch->getCoarseFineInterfaceFaces()->end(); ++iter) {
      Patch::FaceType face = *iter;                   // e.g. xminus=0
      IntVector offset = _patch->faceDirection(face); // e.g. (-1,0,0)
      CellIterator f_iter = _patch->getFaceCellIterator(face,"alongInteriorFaceCells");


      for(; !f_iter.done(); f_iter++) {
        // For each fine cell at C/F interface, compute the index of
        // the neighboring coarse cell (add offset = outward normal to
        // the face and divide by the refinement ratio to obtain
        // "coarseLevel"-level index. Then add the connection between
        // the fine and coarse cells to the graph.
        IntVector fineCell = *f_iter;                        
        IntVector coarseCell = (fineCell + offset) / refRat;

        HYPRE_SStructGraphAddEntries(graph,
                                     fineLevel,fineCell.get_pointer(),
                                     CC_VAR,
                                     coarseLevel,coarseCell.get_pointer(),
                                     CC_VAR);
      }
    }
  } else { // now viewpoint == DoingCoarseToFine
    //==================================================================
    // Add coarse-to-fine connections to graph using the fine-to-coarse
    // connections of fine cells that are stored in A, assuming symmetry.
    //==================================================================
    cout_doing << "Adding coarse-to-fine connections to graph" << "\n";
    const int coarseLevel = _level;
    const int fineLevel = _level+1;
    const IntVector& refRat = grid->getLevel(fineLevel)->getRefinementRatio();
    Level::selectType finePatches;
    _patch->getFineLevelPatches(finePatches); 
    if (finePatches.size() < 1) {
      // No fine patches over me
      return;
    }
    // Loop over all fine patches contained in the coarse patch
    for(int i = 0; i < finePatches.size(); i++){  
      const Patch* finePatch = finePatches[i];
      
      if (finePatch->hasCoarseFineInterfaceFace()) {
        
        // Iterate over coarsefine interface faces
        vector<Patch::FaceType>::const_iterator iter;  
        for (iter  = finePatch->getCoarseFineInterfaceFaces()->begin(); 
             iter != finePatch->getCoarseFineInterfaceFaces()->end(); ++iter) {
          Patch::FaceType face = *iter;                   // e.g. xminus=0
          IntVector offset = _patch->faceDirection(face); // e.g. (-1,0,0)
          CellIterator f_iter = finePatch->getFaceCellIterator(face,"alongInteriorFaceCells");
          
          for(; !f_iter.done(); f_iter++) {
            // For each fine cell at C/F interface, compute the index of
            // the neighboring coarse cell (add offset = outward normal to
            // the face and divide by the refinement ratio to obtain
            // "coarseLevel"-level index. Then add the connection between
            // the fine and coarse cells to the graph.
            IntVector fineCell = *f_iter;                        // in patch
            IntVector coarseCell = (fineCell + offset) / refRat; // outside p.

            HYPRE_SStructGraphAddEntries(graph,
                                         coarseLevel,coarseCell.get_pointer(),
                                         CC_VAR,
                                         fineLevel,fineCell.get_pointer(),
                                         CC_VAR);
          }  // coarse cell interator
        }  // coarseFineInterface faces
      }  // patch has a coarseFineInterface
    }  // finePatch loop 
  } // end if viewpoint == DoingCoarseToFine
} // end HyprePatch_CC::makeConnections(graph)

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
  cout_doing << "Adding interior eqns in patch " << (_patch?_patch->getID():-1)
             << " from "<< _low << " to " << _high << " Level " << _level << "\n";
             
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
    double* values = new double[(_high.x()-_low.x()+1)*stencilSize];
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
    double* values = new double[(_high.x()-_low.x()+1)*stencilSize];
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
    cout_dbg << "Finished interior equation loop" << "\n";
  } 
} // end HyprePatch_CC::makeInteriorConnections()

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
// Add the connections at C/F interfaces of this patch to the HYPRE
// Graph. If viewpoint == DoingFineToCoarse, we add the fine-to-coarse
// connections. If viewpoint == DoingCoarseToFine, we add the
// coarse-to-fine-connections that are read from the connection list
// prepared for this patch by ICE.
//___________________________________________________________________
void
HypreDriverSStruct::HyprePatch_CC::makeConnections(HYPRE_SStructMatrix& HA,
                                                   DataWarehouse* A_dw,
                                                   const VarLabel* A_label,
                                                   const int stencilSize,
                                                   const CoarseFineViewpoint& viewpoint)
{
  // Important: cell iterators here MUST loop in the same order over
  // fine and coarse cells as in makeConnections(graph), otherwise
  // the entry counters (entryFine, entryCoarse) will point to the wrong
  // entries in the Hypre graph.
  const GridP grid = _patch->getLevel()->getGrid();
  const double ZERO = 0.0;
  if (viewpoint == DoingFineToCoarse) {
    //==================================================================
    // Add fine-to-coarse entries to matrix
    //==================================================================
    cout_doing << "Adding fine-to-coarse connections to matrix" << "\n";
    const int fineLevel = _level;
    CCTypes::matrix_type A;
    A_dw->get(A, A_label, _matl, _patch, Ghost::None, 0);
      
    // Keep track of how many connections are added to each coarse
    // cell, because getFaceCellIterator() lexicographically loops
    // over fine cell, so this is not contiguous looping from a single
    // coarse cell's viewpoint.
    CCVariable<int> entryFine;
    A_dw->allocateTemporary(entryFine, _patch);
    // Initialize coutner to # structured entries; no unstructured yet
    entryFine.initialize(stencilSize);

    // Loop over all C/F interface faces
    vector<Patch::FaceType>::const_iterator iter;
    for (iter  = _patch->getCoarseFineInterfaceFaces()->begin(); 
         iter != _patch->getCoarseFineInterfaceFaces()->end(); ++iter) {
      Patch::FaceType face = *iter;                   // e.g. xminus=0
      CellIterator f_iter = _patch->getFaceCellIterator(face,"alongInteriorFaceCells");
      
      cout_dbg << "F/C Face " << face << "\n";
      
      // The equation at fineCell has some unstructured entries numbered
      // stencilSize, stencilSize + 1, ... 
      // (0..stencilSize-1 are the structured entries). We use the value
      // from ICE A[fineCell][face] as the unstructured connection, because
      // it is an approximate flux across the fine face of the C/F boundary.
      const int numGraphEntries = 1;
      const int numStencilEntries = 1;
      int stencilEntries[numStencilEntries] = {face};
      
      for(; !f_iter.done(); f_iter++) {
        IntVector fineCell = *f_iter;                        // inside patch
        int graphEntries[numGraphEntries] = {entryFine[fineCell]};
        const double* graphValues = &A[fineCell][face];
        cout_dbg << "F-C " 
                 << " fLev " << fineLevel << " fCell " << fineCell
                 << " entry " << graphEntries[0]
                 << " value " << graphValues[0];
        HYPRE_SStructMatrixSetValues(HA, _level,
                                     fineCell.get_pointer(),
                                     CC_VAR, numGraphEntries, graphEntries,
                                     const_cast<double*>(graphValues));
        cout_dbg << " done" << "\n";
        entryFine[fineCell]++;    // Update #unstructured connection counter

        // The corresponding structured connection is set to 0.
        const double* stencilValues = &ZERO;
        cout_dbg << "F-F " 
             << " fLev " << _level << " fCell " << fineCell
             << " entry " << stencilEntries[0]
             << " value " << stencilValues[0];
        HYPRE_SStructMatrixSetValues(HA, fineLevel,
                                     fineCell.get_pointer(),
                                     CC_VAR, numStencilEntries, stencilEntries,
                                     const_cast<double*>(stencilValues));
        cout_dbg << " done" << "\n";
      }
    }
  } else { // now viewpoint == DoingCoarseToFine
    //==================================================================
    // Add coarse-to-fine entries to matrix using the fine-to-coarse
    // connections of fine cells that are stored in A, assuming symmetry.
    //==================================================================
    cout_doing << "Adding coarse-to-fine connections to matrix" << "\n";
    const int coarseLevel = _level;
    const int fineLevel = _level+1;
    const LevelP coarse = grid->getLevel(_level);
    const LevelP fine = grid->getLevel(_level+1);
    const IntVector& refRat = fine->getRefinementRatio();
    Level::selectType finePatches;
    _patch->getFineLevelPatches(finePatches); 
    if (finePatches.size() < 1) {
      // No fine patches over me
      return;
    }
    // Keep track of how many connections are added to each coarse
    // cell, because getFaceCellIterator() lexicographically loops
    // over fine cell, so this is not contiguous looping from a single
    // coarse cell's viewpoint.
    CCVariable<int> entryCoarse;
    A_dw->allocateTemporary(entryCoarse, _patch);
    // Initialize coutner to # structured entries; no unstructured yet
    entryCoarse.initialize(stencilSize);

    // Loop over all fine patches contained in the coarse patch
    for(int i = 0; i < finePatches.size(); i++){  
      const Patch* finePatch = finePatches[i];        
      if (finePatch->hasCoarseFineInterfaceFace()) {
        printLine("@",40);
        cout_doing << Parallel::getMPIRank() << " finePatch " << i << " / " << finePatches.size() << "\n"
             << *finePatch << "\n";
        finePatch->printPatchBCs(cout_dbg);
        printLine("@",40);

        IntVector cl, ch, fl, fh;
        getCoarseLevelRange(finePatch, coarse.get_rep(), cl, ch, fl, fh, 1);
        if (fh.x() <= fl.x() || fh.y() <= fl.y() || fh.z() <= fl.z()) {
          continue;
        }

        // Retrieve A of the fine patch, not this patch
        CCTypes::matrix_type AF;
        A_dw->getRegion(AF, A_label, _matl, fine.get_rep(), fl, fh);
        
        // Iterate over C/F interface cells
        vector<Patch::FaceType>::const_iterator iter; 
        for (iter  = finePatch->getCoarseFineInterfaceFaces()->begin(); 
             iter != finePatch->getCoarseFineInterfaceFaces()->end(); ++iter) { 
          Patch::FaceType face = *iter;                   // e.g. xminus=0
          IntVector offset = _patch->faceDirection(face); // e.g. (-1,0,0)
          CellIterator f_iter = finePatch->getFaceCellIterator(face,"alongInteriorFaceCells");

          // check to make sure the face belongs to this coarse patch
          IntVector coarseLow = (f_iter.begin() + offset) / refRat;
          // // add extra to make sure range is >= 1 if it is a valid range
          IntVector coarseHigh = (f_iter.end() + offset) / refRat + Abs(offset); 
          IntVector cl = Max(coarseLow, _patch->getLowIndex());
          IntVector ch = Min(coarseHigh, _patch->getHighIndex());
          if (ch.x() <= cl.x() || ch.y() <= cl.y() || ch.z() <= cl.z()) {
            continue;
          }
          

          // "opposite" = direction opposite to the patch face "face"
          int opposite = face - int(patchFaceSide(face));
          cout_dbg << "C/F Face " << face << " offset " << offset << "\n";
          cout_dbg << "opposite = " << opposite << "\n";
          const int numStencilEntries = 1;
          int stencilEntries[numStencilEntries] = {opposite};
          const double* stencilValues = &ZERO;

          for(; !f_iter.done(); f_iter++) {
            // For each fine cell at C/F interface, compute the index of
            // the neighboring coarse cell (add offset = outward normal to
            // the face and divide by the refinement ratio to obtain
            // "coarseLevel"-level index. Then add the connection between
            // the fine and coarse cells to the graph.
            IntVector fineCell = *f_iter;                        // in patch
            IntVector coarseCell = (fineCell + offset) / refRat; // outside p.
            const int numGraphEntries = 1;
            int graphEntries[numGraphEntries] = {entryCoarse[coarseCell]};
            const double* graphValues = &AF[fineCell][face];
            
            HYPRE_SStructMatrixSetValues(HA, coarseLevel,
                                         coarseCell.get_pointer(),
                                         CC_VAR, numGraphEntries, graphEntries,
                                         const_cast<double*>(graphValues));
                                         
            cout_dbg << "C-F matrix" 
                 << " cLev " << coarseLevel << " cCell " << coarseCell
                 << " (fLev " << fineLevel << " fCell " << fineCell << ")"
                 << " entry " << graphEntries[0];
            cout_dbg << " value " << graphValues[0];                                         
            cout_dbg << " done" << "\n";
            // The corresponding C-C structured connection is already set to 0
            // in impAMRICE.cc.
            if (entryCoarse[coarseCell] == stencilSize) {
              // Hmmm, maybe not. Let's do it outselves. Do it only for one
              // fine cell, hence this if.
              cout_dbg << "C-C " 
                   << " cLev " << _level << " cCell " << coarseCell
                   << " entry " << stencilEntries[0]
                   << " value " << stencilValues[0];
              HYPRE_SStructMatrixSetValues(HA, _level,
                                           coarseCell.get_pointer(),
                                           CC_VAR, numStencilEntries, stencilEntries,
                                           const_cast<double*>(stencilValues));
              cout_dbg << " done" << "\n";
            } // end if first fine child

            entryCoarse[coarseCell]++;    // Update #unstructured connection counter
          }  // end fine cell interator
        }  // coarseFineInterface faces
      }  // patch has a coarseFineInterface
    }  // finePatch loop 
  } // end if viewpoint == DoingCoarseToFine
} // end HyprePatch_CC::makeConnections(matrix)

//___________________________________________________________________
// getSolution(): Move the solution from hypre to uintah
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

//#####################################################################
// Utilities
//#####################################################################

void printLine(const string& s, const unsigned int len)
{
  for (unsigned int i = 0; i < len; i++) {
    cout_dbg << s;
  }
  cout_dbg << "\n";
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
