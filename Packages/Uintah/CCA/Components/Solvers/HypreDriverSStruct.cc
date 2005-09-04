/*--------------------------------------------------------------------------
 * File: HypreDriverSStruct.cc
 *
 * Implementation of a wrapper of a Hypre solvers working with the Hypre
 * SStruct system interface.
 *--------------------------------------------------------------------------*/

#include <sci_defs/hypre_defs.h>

#if HAVE_HYPRE_1_9
#include <Packages/Uintah/CCA/Components/Solvers/HypreDriverSStruct.h>
#include <Packages/Uintah/CCA/Components/Solvers/MatrixUtil.h>

#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Packages/Uintah/Core/Grid/Variables/Stencil7.h>
#include <Packages/Uintah/Core/Grid/Variables/VarTypes.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>

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
{}

HypreDriverSStruct::HyprePatch::~HyprePatch(void)
  //___________________________________________________________________
  // HypreDriverSStruct::HyprePatch destructor
  //___________________________________________________________________
{}

HypreDriverSStruct::~HypreDriverSStruct(void)
  //___________________________________________________________________
  // HypreDriverSStruct destructor
  //___________________________________________________________________
{
  cerr << "HypreDriverSStruct destructor BEGIN" << "\n";
  printDataStatus();
  // Destroy matrix, RHS, solution objects
  cerr << "Destroying SStruct matrix, RHS, solution objects" << "\n";
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
  cerr << "Destroying graph object" << "\n";
  if (_exists[SStructGraph] >= SStructAssembled) {
    HYPRE_SStructGraphDestroy(_graph);
    _exists[SStructGraph] = SStructDestroyed;
  }

  // Destroying grid, stencil
  if (_exists[SStructStencil] >= SStructCreated) {
    cerr << "Destroying stencil object" << "\n";
    HYPRE_SStructStencilDestroy(_stencil);
    _exists[SStructStencil] = SStructDestroyed;
  }
  if (_vars) {
    delete _vars;
  }
  if (_exists[SStructGrid] >= SStructAssembled) {
    cerr << "Destroying grid object" << "\n";
    HYPRE_SStructGridDestroy(_grid);
    _exists[SStructGrid] = SStructDestroyed;
  }
  cerr << "HypreDriverSStruct destructor END" << "\n";
}

void
HypreDriverSStruct::printMatrix(const string& fileName /* =  "output" */)
{
  cerr << "HypreDriverSStruct::printMatrix() begin" << "\n";
  if (!_params->printSystem) return;
  HYPRE_SStructMatrixPrint((fileName + ".sstruct").c_str(), _HA, 0);
  if (_requiresPar) {
    HYPRE_ParCSRMatrixPrint(_HA_Par, (fileName + ".par").c_str());
    // Print CSR matrix in IJ format, base 1 for rows and cols
    HYPRE_ParCSRMatrixPrintIJ(_HA_Par, 1, 1, (fileName + ".ij").c_str());
  }
  cerr << "HypreDriverSStruct::printMatrix() end" << "\n";
}

void
HypreDriverSStruct::printRHS(const string& fileName /* =  "output_b" */)
{
  if (!_params->printSystem) return;
  HYPRE_SStructVectorPrint(fileName.c_str(), _HB, 0);
  if (_requiresPar) {
    HYPRE_ParVectorPrint(_HB_Par, (fileName + ".par").c_str());
  }
}

void
HypreDriverSStruct::printSolution(const string& fileName /* =  "output_x" */)
{
  if (!_params->printSystem) return;
  HYPRE_SStructVectorPrint(fileName.c_str(), _HX, 0);
  if (_requiresPar) {
    HYPRE_ParVectorPrint(_HX_Par, (fileName + ".par").c_str());
  }
}

void
HypreDriverSStruct::gatherSolutionVector(void)
{
  HYPRE_SStructVectorGather(_HX);
} // end HypreDriverSStruct::gatherSolutionVector()

void
HypreDriverSStruct::printDataStatus(void)
{
  cerr << "Hypre SStruct interface data status:" << "\n";
  for (unsigned i = 0; i < _exists.size(); i++) {
    cerr << "_exists[" << i << "] = " << _exists[i] << "\n";
  }
}

//#####################################################################
// class HypreDriverSStruct implementation for CC variable type
//#####################################################################

static const int CC_NUM_VARS = 1; // # Hypre var types that we use in CC solves
static const int CC_VAR = 0;      // Hypre CC variable type index

void
HypreDriverSStruct::makeLinearSystem_CC(const int matl)
  //___________________________________________________________________
  // Function HypreDriverSStruct::makeLinearSystem_CC~
  // Construct the linear system for CC variables (e.g. pressure),
  // for the Hypre Struct interface (1-level problem / uniform grid).
  // We set up the matrix at all patches of the "level" data member.
  // matl is a fake material index. We always have one material here,
  // matl=0 (pressure).
  //___________________________________________________________________
{
  cerr << "HypreDriverSStruct::makeLinearSystem_CC() BEGIN" << "\n";
  ASSERTEQ(sizeof(Stencil7), 7*sizeof(double));
  //==================================================================
  // Set up the grid
  //==================================================================
  cerr << "Setting up the SStruct grid" << "\n";
  // Create an empty grid in 3 dimensions with # parts = numLevels.
  const int numDims = 3;
  const int numLevels = _level->getGrid()->numLevels();
  HYPRE_SStructGridCreate(_pg->getComm(), numDims, numLevels, &_grid);
  cerr << "Constructed empty grid, numDims " << numDims
       << " numParts " << numLevels << "\n";
  _exists[SStructGrid] = SStructCreated;
  _vars = new HYPRE_SStructVariable[CC_NUM_VARS];
  _vars[CC_VAR] = HYPRE_SSTRUCT_VARIABLE_CELL; // We use only cell centered var

  // Loop over the Uintah patches that this proc owns
  for (int p = 0 ; p < _patches->size(); p++) {
    HyprePatch_CC hpatch(_patches->get(p),matl); // Read Uintah patch data
    hpatch.addToGrid(_grid,_vars);
  }
  HYPRE_SStructGridAssemble(_grid);
  _exists[SStructGrid] = SStructAssembled;

  //==================================================================
  // Set up the stencil
  //==================================================================
  cerr << "Setting up the SStruct stencil" << "\n";
  // Prepare index offsets and stencil size
  cerr << "Symmetric = " << _params->symmetric << "\n";
  if (_params->symmetric) {
    _stencilSize = numDims+1;
    int offsets[4][numDims] = {{0,0,0},
                               {-1,0,0},
                               {0,-1,0},
                               {0,0,-1}};
    // Feed offsets into stencil
    HYPRE_SStructStencilCreate(numDims, _stencilSize, &_stencil);
    for (int i = 0; i < _stencilSize; i++) {
      HYPRE_SStructStencilSetEntry(_stencil, i, offsets[i], 0);
    }
  } else {
    _stencilSize = 2*numDims+1;
    int offsets[7][numDims] = {{0,0,0},
                               {1,0,0}, {-1,0,0},
                               {0,1,0}, {0,-1,0},
                               {0,0,1}, {0,0,-1}};
    // Feed offsets into stencil
    HYPRE_SStructStencilCreate(numDims, _stencilSize, &_stencil);
    for (int i = 0; i < _stencilSize; i++) {
      HYPRE_SStructStencilSetEntry(_stencil, i, offsets[i], 0);
    }
  }
  _exists[SStructStencil] = SStructCreated;

  //==================================================================
  // Set up the SStruct unstructured connection graph _graph
  //==================================================================
  cerr << "Setting up the SStruct graph" << "\n";
  // Create an empty graph
  HYPRE_SStructGraphCreate(_pg->getComm(), _grid, &_graph);
  _exists[SStructGraph] = SStructCreated;
  // For ParCSR-requiring solvers like AMG
  if (_requiresPar) {
    cerr << "graph object type set to HYPRE_PARCSR" << "\n";
    HYPRE_SStructGraphSetObjectType(_graph, HYPRE_PARCSR);
  }

  //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
  //  Add structured equations (stencil-based) at the interior of
  //  each patch at every level to the graph.
  //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
  printLine("*",50);
  cerr << "Graph structured (interior) connections" << "\n";
  printLine("*",50);
  for (int level = 0; level < numLevels; level++) {
    cerr << "  Initializing graph stencil at level " << level
         << " of " << numLevels << "\n";
    HYPRE_SStructGraphSetStencil(_graph, level, CC_VAR, _stencil);
  }

  //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
  // Add to graph the unstructured part of the stencil connecting the
  // coarse and fine level at every C/F boundary (F->C connections at
  // this patch's outer boundaries, and C->F connections at all
  // applicable C/F boundaries of next-finer-level patches that lie
  // above this patch.
  //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
  printLine("*",50);
  cerr << "Graph unstructured connections" << "\n";
  printLine("*",50);

  // Add Uintah patches that this proc owns
  for (int p = 0; p < _patches->size(); p++) {
    // Read Uintah patch info into our data structure
    const Patch* patch = _patches->get(p);
    HyprePatch_CC hpatch(_patches->get(p),matl); // Read Uintah patch data
    int level = hpatch.getLevel();
    printLine("$",40);
    cerr << "Processing Patch" << "\n" << hpatch << "\n";
    patch->printPatchBCs(cerr);
    printLine("$",40);
    
    if ((level > 0) && (patch->hasCoarseFineInterfaceFace())) {
      // If not at coarsest level, add fine-to-coarse connections at all
      // C/F interface faces.
      printLine("=",50);
      cerr << "Building fine-to-coarse connections" << "\n";
      printLine("=",50);
      hpatch.makeConnections(_graph,DoingFineToCoarse);
    } // end if (level > 0) and (patch has a CFI)

    if (level < numLevels-1) {
      // If not at finest level, examine the connection list that
      // impAMRICE.cc provides us and add the coarse-to-fine
      // connections to the Hypre graph. This list should cover all
      // the C/F interfaces of all next-finer level patches inscribed
      // in this patch.
      printLine("=",50);
      cerr << "Building coarse-to-fine connections" << "\n";
      printLine("=",50);
      hpatch.makeConnections(_graph,DoingCoarseToFine);
    } // end if (level < numLevels-1)
  } // end for p (patches)
  
  // Assemble the graph
  HYPRE_SStructGraphAssemble(_graph);
  _exists[SStructGraph] = SStructAssembled;
  cerr << "Assembled graph, nUVentries = "
       << hypre_SStructGraphNUVEntries(_graph) << "\n";

  //==================================================================
  // Set up the Struct left-hand-side matrix _HA
  //==================================================================
  cerr << "Setting up the SStruct matrix _HA" << "\n";
  // Create and initialize an empty SStruct matrix
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
    cerr << "HA object type set to HYPRE_PARCSR" << "\n";
    HYPRE_SStructMatrixSetObjectType(_HA, HYPRE_PARCSR);
  }
  HYPRE_SStructMatrixInitialize(_HA);
  _exists[SStructA] = SStructInitialized;

  //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
  // Add structured equation (stencil-based) entries at the interior of
  // each patch at every level to the matrix.
  //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
  printLine("*",50);
  cerr << "Matrix structured (interior) entries" << "\n";
  printLine("*",50);
  for (int p = 0 ; p < _patches->size(); p++) {
    // Read Uintah patch info into our data structure, set Uintah pointers
    const Patch* patch = _patches->get(p);
    HyprePatch_CC hpatch(patch,matl); // Read Uintah patch data
    hpatch.makeInteriorEquations(_HA, _A_dw, _A_label,
                                 _stencilSize, _params->symmetric);
  } // end for p (patches)

  //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
  // Add/update unstructured equation entries at C/F interfaces 
  // each patch at every level to the matrix.
  //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
  printLine("*",50);
  cerr << "Matrix unstructured (C/F) entries" << "\n";
  printLine("*",50);

  // Add Uintah patches that this proc owns
  for (int p = 0; p < _patches->size(); p++) {
    // Read Uintah patch info into our data structure
    const Patch* patch = _patches->get(p);
    HyprePatch_CC hpatch(_patches->get(p),matl); // Read Uintah patch data
    int level = hpatch.getLevel();
    printLine("$",40);
    cerr << "Processing Patch" << "\n" << hpatch << "\n";
    printLine("$",40);
    
    if ((level > 0) && (patch->hasCoarseFineInterfaceFace())) {
      // If not at coarsest level, add fine-to-coarse connections at all
      // C/F interface faces.
      printLine("=",50);
      cerr << "Building fine-to-coarse entries" << "\n";
      printLine("=",50);
      patch->printPatchBCs(cerr);
      hpatch.makeConnections(_HA, _A_dw, _A_label,
                             _stencilSize, DoingFineToCoarse);
    } // end if (level > 0) and (patch has a CFI)

    if (level < numLevels-1) {
      // If not at finest level, examine the connection list that
      // impAMRICE.cc provides us and add the coarse-to-fine
      // connections to the Hypre graph. This list should cover all
      // the C/F interfaces of all next-finer level patches inscribed
      // in this patch.
      printLine("=",50);
      cerr << "Building coarse-to-fine entries" << "\n";
      printLine("=",50);
      hpatch.makeConnections(_HA, _A_dw, _A_label,
                             _stencilSize, DoingCoarseToFine);
    } // end if (level < numLevels-1)
  } // end for p (patches)

  // This is an all-proc collective call
  HYPRE_SStructMatrixAssemble(_HA);
  _exists[SStructA] = SStructAssembled;

  //==================================================================
  // Set up the Struct right-hand-side vector _HB
  //==================================================================
  cerr << "Setting up the SStruct RHS vector _HB" << "\n";
  HYPRE_SStructVectorCreate(_pg->getComm(), _grid, &_HB);
  _exists[SStructB] = SStructCreated;
  // For solvers that require ParCSR format
  if (_requiresPar) {
    cerr << "HB object type set to HYPRE_PARCSR" << "\n";
    HYPRE_SStructVectorSetObjectType(_HB, HYPRE_PARCSR);
  }
  HYPRE_SStructVectorInitialize(_HB);
  _exists[SStructB] = SStructInitialized;

  // Set RHS vector entries at the interior of
  // each patch at every level to the matrix.
  printLine("*",50);
  cerr << "Set RHS vector entries" << "\n";
  printLine("*",50);
  for (int p = 0 ; p < _patches->size(); p++) {
    // Read Uintah patch info into our data structure, set Uintah pointers
    const Patch* patch = _patches->get(p);
    HyprePatch_CC hpatch(patch,matl); // Read Uintah patch data
    hpatch.makeInteriorVector(_HB, _b_dw, _B_label);
  } // end for p (patches)

  HYPRE_SStructVectorAssemble(_HB);
  _exists[SStructB] = SStructAssembled;

  //==================================================================
  // Set up the Struct solution vector _HX
  //==================================================================
  cerr << "Setting up the SStruct solution vector _HX" << "\n";
  HYPRE_SStructVectorCreate(_pg->getComm(), _grid, &_HX);
  _exists[SStructX] = SStructCreated;
  // For solvers that require ParCSR format
  if (_requiresPar) {
    cerr << "HX object type set to HYPRE_PARCSR" << "\n";
    HYPRE_SStructVectorSetObjectType(_HX, HYPRE_PARCSR);
  }
  HYPRE_SStructVectorInitialize(_HX);
  _exists[SStructX] = SStructInitialized;

  // Set solution (initial guess) vector entries at the interior of
  // each patch at every level to the matrix.
  printLine("*",50);
  cerr << "Set solution vector initial guess entries" << "\n";
  printLine("*",50);
  if (_guess_label) {
    // If guess is provided by ICE, read it from Uintah
    cerr << "Reading initial guess from ICE" << "\n";
    for (int p = 0 ; p < _patches->size(); p++) {
      // Read Uintah patch info into our data structure, set Uintah pointers
      const Patch* patch = _patches->get(p);
      HyprePatch_CC hpatch(patch,matl); // Read Uintah patch data
      hpatch.makeInteriorVector(_HX, _guess_dw, _guess_label);
    } // end for p (patches)
  } else {
#if 0
    // If guess is not provided by ICE, use zero as initial guess
    cerr << "Default initial guess: zero" << "\n";
    for (int p = 0 ; p < _patches->size(); p++) {
      // Read Uintah patch info into our data structure, set Uintah pointers
      const Patch* patch = _patches->get(p);
      HyprePatch_CC hpatch(patch,matl); // Read Uintah patch data
      hpatch.makeInteriorVectorZero(_HX, _guess_dw, _guess_label);
    } // end for p (patches)
#endif
  }

  HYPRE_SStructVectorAssemble(_HX);
  _exists[SStructX] = SStructAssembled;

  // For solvers that require ParCSR format
  if (_requiresPar) {
    cerr << "Making ParCSR objects from SStruct objects" << "\n";
    HYPRE_SStructMatrixGetObject(_HA, (void **) &_HA_Par);
    HYPRE_SStructVectorGetObject(_HB, (void **) &_HB_Par);
    HYPRE_SStructVectorGetObject(_HX, (void **) &_HX_Par);
  }
  cerr << "HypreDriverSStruct::makeLinearSystem_CC() END" << "\n";
} // end HypreDriverSStruct::makeLinearSystem_CC()


void
HypreDriverSStruct::getSolution_CC(const int matl)
  //_____________________________________________________________________
  // Function HypreDriverSStruct::getSolution_CC~
  // Get the solution vector for a multi-level, CC variable problem from
  // the Hypre SStruct interface.
  //_____________________________________________________________________*/
{
  // Loop over the Uintah patches that this proc owns
  for (int p = 0 ; p < _patches->size(); p++) {
    //==================================================================
    // Find patch extents and level it belongs to
    //==================================================================
    HyprePatch_CC hpatch(_patches->get(p),matl); // Read Uintah patch data
    hpatch.getSolution(_HX,_new_dw,_X_label,_modifies_x);
  } // for p (patches)
} // end HypreDriverSStruct::getSolution_CC()

void
HypreDriverSStruct::HyprePatch_CC::addToGrid
(HYPRE_SStructGrid& grid,
 HYPRE_SStructVariable* vars)
  //___________________________________________________________________
  // Add this patch to the Hypre grid
  //___________________________________________________________________
{
  cerr << "Adding patch " << _patch->getID()
       << " from "<< _low << " to " << _high
       << " Level " << _level << "\n";
  HYPRE_SStructGridSetExtents(grid, _level,
                              _low.get_pointer(),
                              _high.get_pointer());
  HYPRE_SStructGridSetVariables(grid, _level, CC_NUM_VARS, vars);
}

void
HypreDriverSStruct::HyprePatch_CC::makeConnections
(HYPRE_SStructGraph& graph,
 const CoarseFineViewpoint& viewpoint)
  //___________________________________________________________________
  // HypreDriverSStruct::HyprePatch_CC::makeConnections~
  // Add the connections at C/F interfaces of this patch to the HYPRE
  // Graph. If viewpoint == DoingFineToCoarse, we add the fine-to-coarse
  // connections. If viewpoint == DoingCoarseToFine, we add the
  // coarse-to-fine-connections that are read from the connection list
  // prepared for this patch by ICE.
  //___________________________________________________________________
{
  cerr << "Doing makeConnections graph for Patch "
       << _patch->getID() 
       << " Level " << _level << "\n";
  const GridP grid = _patch->getLevel()->getGrid();
  if (viewpoint == DoingFineToCoarse) {
    //==================================================================
    // Add fine-to-coarse connections to graph
    //==================================================================
    cerr << "Adding fine-to-coarse connections to graph" << "\n";
    const int fineLevel = _level;
    const int coarseLevel = _level-1;
    const IntVector& refRat = grid->getLevel(fineLevel)->getRefinementRatio();
    vector<Patch::FaceType>::const_iterator iter;  
    // Loop over all C/F interface faces (note: a Uintah patch face is
    // marked as a coarsefine interface only when the current patch is
    // the FINE patch (what Todd calls "looking up is easy, not
    // looking down").  See also ICE/impAMRICE.cc,
    // ICE::matrixBC_CFI_finePatch().
    for (iter = _patch->getCoarseFineInterfaceFaces()->begin(); 
         iter != _patch->getCoarseFineInterfaceFaces()->end(); ++iter) {
      Patch::FaceType face = *iter;                   // e.g. xminus=0
      IntVector offset = _patch->faceDirection(face); // e.g. (-1,0,0)
      CellIterator f_iter = 
        _patch->getFaceCellIterator(face,"alongInteriorFaceCells");
      cerr << "F/C Face " << face << " offset " << offset << "\n";
      for(; !f_iter.done(); f_iter++) {
        // For each fine cell at C/F interface, compute the index of
        // the neighboring coarse cell (add offset = outward normal to
        // the face and divide by the refinement ratio to obtain
        // "coarseLevel"-level index. Then add the connection between
        // the fine and coarse cells to the graph.
        IntVector fineCell = *f_iter;                        // inside patch
        IntVector coarseCell = (fineCell + offset) / refRat; // outside patch
        cerr << "F-C entry"
             << " fLev " << fineLevel << " fCell " << fineCell
             << " cLev " << coarseLevel << " cCell " << coarseCell;
        HYPRE_SStructGraphAddEntries(graph,
                                     fineLevel,fineCell.get_pointer(),
                                     CC_VAR,
                                     coarseLevel,coarseCell.get_pointer(),
                                     CC_VAR);
        cerr << " done" << "\n";
      }
    }
  } else { // now viewpoint == DoingCoarseToFine
    //==================================================================
    // Add coarse-to-fine connections to graph using the fine-to-coarse
    // connections of fine cells that are stored in A, assuming symmetry.
    //==================================================================
    cerr << "Adding coarse-to-fine connections to graph" << "\n";
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
        printLine("@",40);
        cerr << "finePatch " << i << " / " << finePatches.size() << "\n"
             << *finePatch << "\n";
        finePatch->printPatchBCs(cerr);
        printLine("@",40);
        // Iterate over coarsefine interface faces
        vector<Patch::FaceType>::const_iterator iter;  
        for (iter  = finePatch->getCoarseFineInterfaceFaces()->begin(); 
             iter != finePatch->getCoarseFineInterfaceFaces()->end(); ++iter) {
          Patch::FaceType face = *iter;                   // e.g. xminus=0
          IntVector offset = _patch->faceDirection(face); // e.g. (-1,0,0)
          CellIterator f_iter = 
            finePatch->getFaceCellIterator(face,"alongInteriorFaceCells");
          cerr << "C/F Face " << face << " offset " << offset << "\n";
          for(; !f_iter.done(); f_iter++) {
            // For each fine cell at C/F interface, compute the index of
            // the neighboring coarse cell (add offset = outward normal to
            // the face and divide by the refinement ratio to obtain
            // "coarseLevel"-level index. Then add the connection between
            // the fine and coarse cells to the graph.
            IntVector fineCell = *f_iter;                        // in patch
            IntVector coarseCell = (fineCell + offset) / refRat; // outside p.
            cerr << "C-F graph" 
                 << " cLev " << coarseLevel << " cCell " << coarseCell
                 << " fLev " << fineLevel << " fCell " << fineCell;
            HYPRE_SStructGraphAddEntries(graph,
                                         coarseLevel,coarseCell.get_pointer(),
                                         CC_VAR,
                                         fineLevel,fineCell.get_pointer(),
                                         CC_VAR);
            cerr << " done" << "\n";
          }  // coarse cell interator
        }  // coarseFineInterface faces
      }  // patch has a coarseFineInterface
    }  // finePatch loop 
  } // end if viewpoint == DoingCoarseToFine
} // end HyprePatch_CC::makeConnections(graph)

void
HypreDriverSStruct::HyprePatch_CC::makeInteriorEquations
(HYPRE_SStructMatrix& HA,
 DataWarehouse* A_dw,
 const VarLabel* A_label,
 const int stencilSize,
 const bool symmetric /* = false */)
  //___________________________________________________________________
  // HypreDriverSStruct::HyprePatch_CC::makeInteriorEquations~
  // Add the connections at C/F interfaces of this patch to the HYPRE
  // Graph. If viewpoint == DoingFineToCoarse, we add the fine-to-coarse
  // connections. If viewpoint == DoingCoarseToFine, we add the
  // coarse-to-fine-connections that are read from the connection list
  // prepared for this patch by ICE.
  //___________________________________________________________________
{
  cerr << "Adding interior eqns in patch " << _patch->getID()
       << " from "<< _low << " to " << _high
       << " Level " << _level << "\n";
  CCTypes::matrix_type A;
  A_dw->get(A, A_label, _matl, _patch, Ghost::None, 0);
  if (symmetric) {
    //==================================================================
    // Add symmetric stencil equations to HA
    //==================================================================
    // Because AA is 7-point and the stencil is 4-point, copy data from AA
    // into stencil, and then feed it to Hypre
    double* values =
      new double[(_high.x()-_low.x()+1)*stencilSize];
    int stencil_indices[] = {0,1,2,3};
    cerr << "Reading 4-point stencil from ICE::A" << "\n";
    cerr << "stencilSize = " << stencilSize << "\n";
    for(int z = _low.z(); z <= _high.z(); z++) {
      for(int y = _low.y(); y <= _high.y(); y++) {
        // Read data in "chunks" of fixed y-, z- index and running x-index
        const Stencil7* AA = &A[IntVector(_low.x(), y, z)];
        double* p = values;
        for (int x = _low.x(); x <= _high.x(); x++) {
          *p++ = AA->p;
          *p++ = AA->w;
          *p++ = AA->s;
          *p++ = AA->b;
          AA++;
        }
        IntVector chunkLow(_low.x(), y, z);
        IntVector chunkHigh(_high.x(), y, z);
        // Feed data from Uintah to Hypre
        HYPRE_SStructMatrixSetBoxValues(HA, _level,
                                        chunkLow.get_pointer(),
                                        chunkHigh.get_pointer(), CC_VAR,
                                        stencilSize, stencil_indices, values);
      }
    }
    delete[] values;
  } else { // now symmetric = false
    //==================================================================
    // Add non-symmetric stencil equations to HA
    //==================================================================
#if 0
    // Taken from Steve's HypreSolver code, but seems dubious, because
    // p is the last, not first data member of Stencil7, so pointing to
    // it and incrementing the pointer to retrieve indices 0,1,2,3,4,5,6
    // seems out of order.

    // AA is 7-point and stencil is 7-point, feed data directly to Hypre
    cerr << "Reading 7-point stencil from ICE::A" << "\n";
    cerr << "stencilSize = " << stencilSize << "\n";
    int stencil_indices[] = {0,1,2,3,4,5,6};
    for(int z = _low.z(); z <= _high.z(); z++) {
      for(int y = _low.y(); y <= _high.y(); y++) {
        const double* values = &A[IntVector(_low.x(), y, z)].p;
        IntVector chunkLow(_low.x(), y, z);
        IntVector chunkHigh(_high.x(), y, z);
        // Feed data from Uintah to Hypre
        HYPRE_SStructMatrixSetBoxValues(HA, _level,
                                        chunkLow.get_pointer(),
                                        chunkHigh.get_pointer(), CC_VAR,
                                        stencilSize, stencil_indices,
                                        const_cast<double*>(values));
      }
    }
#else
    // Use similar code to the symmetric case, with specific points to 
    // data members of Stencil7, not relying on their ordering.
    double* values =
      new double[(_high.x()-_low.x()+1)*stencilSize];
    int stencil_indices[] = {0,1,2,3,4,5,6};
    cerr << "'Manually' Reading 7-point stencil from ICE::A" << "\n";
    cerr << "stencilSize = " << stencilSize << "\n";
    cerr << "_low        = " << _low << "\n";
    cerr << "_high       = " << _high << "\n";
    for(int z = _low.z(); z <= _high.z(); z++) {
      for(int y = _low.y(); y <= _high.y(); y++) {
        // Read data in "chunks" of fixed y-, z- index and running x-index
        const Stencil7* AA = &A[IntVector(_low.x(), y, z)];
        double* p = values;
        for (int x = _low.x(); x <= _high.x(); x++) {
          *p++ = AA->p;
          *p++ = AA->e;
          *p++ = AA->w;
          *p++ = AA->n;
          *p++ = AA->s;
          *p++ = AA->t;
          *p++ = AA->b;
          cerr << "Stencil at (" << x << "," << y << "," << z << ") = " << "\n";
          cerr << "p " << AA->p
               << " e " << AA->e
               << " w " << AA->w
               << " n " << AA->n
               << " s " << AA->s
               << " t " << AA->t
               << " b " << AA->b
               << "\n";
          AA++;
        }
        IntVector chunkLow(_low.x(), y, z);
        IntVector chunkHigh(_high.x(), y, z);
        // Feed data from Uintah to Hypre
        HYPRE_SStructMatrixSetBoxValues(HA, _level,
                                        chunkLow.get_pointer(),
                                        chunkHigh.get_pointer(), CC_VAR,
                                        stencilSize, stencil_indices, values);
      }
    }
    delete[] values;
    cerr << "Finished interior equation loop" << "\n";
#endif
  } // end if (symmetric) 
} // end HyprePatch_CC::makeInteriorConnections()

void
HypreDriverSStruct::HyprePatch_CC::makeInteriorVector
(HYPRE_SStructVector& HV,
 DataWarehouse* V_dw,
 const VarLabel* V_label)
  //___________________________________________________________________
  // HypreDriverSStruct::HyprePatch_CC::makeInteriorVector~
  // Read the vector HV from Uintah into Hypre. HV can be the RHS
  // or the solution (initial guess) vector. HV is defined at the interior
  // cells of each patch.
  //___________________________________________________________________
{
  CCTypes::const_type V;
  V_dw->get(V, V_label, _matl, _patch, Ghost::None, 0);
  for(int z = _low.z(); z <= _high.z(); z++) {
    for(int y = _low.y(); y <= _high.y(); y++) {
      const double* values = &V[IntVector(_low.x(), y, z)];
      IntVector chunkLow(_low.x(), y, z);
      IntVector chunkHigh(_high.x(), y, z);
      // Feed data from Uintah to Hypre
      HYPRE_SStructVectorSetBoxValues(HV, _level,
                                      chunkLow.get_pointer(),
                                      chunkHigh.get_pointer(), CC_VAR,
                                      const_cast<double*>(values));
    }
  }
} // end HyprePatch_CC::makeInteriorVector()

void
HypreDriverSStruct::HyprePatch_CC::makeInteriorVectorZero
(HYPRE_SStructVector& HV,
 DataWarehouse* V_dw,
 const VarLabel* V_label)
  //___________________________________________________________________
  // HypreDriverSStruct::HyprePatch_CC::makeInteriorVector~
  // Read the vector HV from Uintah into Hypre. HV can be the RHS
  // or the solution (initial guess) vector. HV is defined at the interior
  // cells of each patch.
  //___________________________________________________________________
{
#if 0
  // Make a vector of zeros
  // Is this a way to create a vector of zeros in Uintah?
  CCVariable<double> V;
  V_dw->allocateAndPut(V, V_label, _matl, _patch);
  V.initialize(0.0);

  // Or that?
  CCTypes::const_type V;
  V_dw->get(V, V_label, _matl, _patch, Ghost::None, 0);
  for(int z = _low.z(); z <= _high.z(); z++) {
    for(int y = _low.y(); y <= _high.y(); y++) {
      const double* values = &V[IntVector(_low.x(), y, z)];
      IntVector chunkLow(_low.x(), y, z);
      IntVector chunkHigh(_high.x(), y, z);
      // Feed data from Uintah to Hypre
      HYPRE_SStructVectorSetBoxValues(HV, _level,
                                      chunkLow.get_pointer(),
                                      chunkHigh.get_pointer(), CC_VAR,
                                      const_cast<double*>(values));
    }
  }
#endif
} // end HyprePatch_CC::makeInteriorVector()

void
HypreDriverSStruct::HyprePatch_CC::makeConnections
(HYPRE_SStructMatrix& HA,
 DataWarehouse* A_dw,
 const VarLabel* A_label,
 const int stencilSize,
 const CoarseFineViewpoint& viewpoint)
  //___________________________________________________________________
  // HypreDriverSStruct::HyprePatch_CC::makeConnections~
  // Add the connections at C/F interfaces of this patch to the HYPRE
  // Graph. If viewpoint == DoingFineToCoarse, we add the fine-to-coarse
  // connections. If viewpoint == DoingCoarseToFine, we add the
  // coarse-to-fine-connections that are read from the connection list
  // prepared for this patch by ICE.
  //___________________________________________________________________
{
  const GridP grid = _patch->getLevel()->getGrid();
  const double ZERO = 0.0;
  if (viewpoint == DoingFineToCoarse) {
    //==================================================================
    // Add fine-to-coarse entries to matrix
    //==================================================================
    const int fineLevel = _level;
    CCTypes::matrix_type A;
    A_dw->get(A, A_label, _matl, _patch, Ghost::None, 0);
    vector<Patch::FaceType>::const_iterator iter;  
    // Loop over all C/F interface faces
    for (iter  = _patch->getCoarseFineInterfaceFaces()->begin(); 
         iter != _patch->getCoarseFineInterfaceFaces()->end(); ++iter) {
      Patch::FaceType face = *iter;                   // e.g. xminus=0
      CellIterator f_iter =
        _patch->getFaceCellIterator(face,"alongInteriorFaceCells");
      cerr << "F/C Face " << face << "\n";
      // The equation at fineCell has one unstructured entry no. stencilSize
      // (0..stencilSize-1 are the structured entries). We use the value
      // from ICE A[fineCell][face] as the unstructured connection, because
      // it is an approximate flux across the fine face of the C/F boundary.
      const int numGraphEntries = 1;
      int graphEntries[numGraphEntries] = {stencilSize};
      const int numStencilEntries = 1;
      int stencilEntries[numStencilEntries] = {face};
      for(; !f_iter.done(); f_iter++) {
        IntVector fineCell = *f_iter;                        // inside patch
        const double* graphValues = &A[fineCell][face];
        cerr << "F-C " 
             << " fLev " << fineLevel << " fCell " << fineCell
             << " entry " << graphEntries[0]
             << " value " << graphValues[0];
        HYPRE_SStructMatrixSetValues(HA, _level,
                                     fineCell.get_pointer(),
                                     CC_VAR, numGraphEntries, graphEntries,
                                     const_cast<double*>(graphValues));
        cerr << " done" << "\n";

        // The corresponding structured connection is set to 0.
        const double* stencilValues = &ZERO;
        cerr << "F-F " 
             << " fLev " << _level << " fCell " << fineCell
             << " entry " << stencilEntries[0]
             << " value " << stencilValues[0];
        HYPRE_SStructMatrixSetValues(HA, fineLevel,
                                     fineCell.get_pointer(),
                                     CC_VAR, numStencilEntries, stencilEntries,
                                     const_cast<double*>(stencilValues));
        cerr << " done" << "\n";
      }
    }
  } else { // now viewpoint == DoingCoarseToFine
    //==================================================================
    // Add coarse-to-fine entries to matrix using the fine-to-coarse
    // connections of fine cells that are stored in A, assuming symmetry.
    //==================================================================
    cerr << "Adding coarse-to-fine connections to graph" << "\n";
    const int coarseLevel = _level;
    const int fineLevel = _level+1;
    const IntVector& refRat = grid->getLevel(fineLevel)->getRefinementRatio();
    Level::selectType finePatches;
    _patch->getFineLevelPatches(finePatches); 
    if (finePatches.size() < 1) {
      // No fine patches over me
      return;
    }
    // Keeps track of how many connections are added to each coarse
    // cell, because getFaceCellIterator() lexicographically loops
    // over fine cell, so this is not contiguous looping from a single
    // coarse cell's viewpoint.
    CCVariable<int> entry;
    A_dw->allocateTemporary(entry, _patch);
    entry.initialize(stencilSize); // # structured entries; no unstructured yet

    // Loop over all fine patches contained in the coarse patch
    for(int i = 0; i < finePatches.size(); i++){  
      const Patch* finePatch = finePatches[i];        
      if (finePatch->hasCoarseFineInterfaceFace()) {
        printLine("@",40);
        cerr << "finePatch " << i << " / " << finePatches.size() << "\n"
             << *finePatch << "\n";
        finePatch->printPatchBCs(cerr);
        printLine("@",40);
        // Retrieve A of the fine patch, not this patch
        CCTypes::matrix_type AF;
        A_dw->get(AF, A_label, _matl, finePatch, Ghost::None, 0);
        // Iterate over coarsefine interface faces
        vector<Patch::FaceType>::const_iterator iter; 
        for (iter  = finePatch->getCoarseFineInterfaceFaces()->begin(); 
             iter != finePatch->getCoarseFineInterfaceFaces()->end(); ++iter) {
          Patch::FaceType face = *iter;                   // e.g. xminus=0
          IntVector offset = _patch->faceDirection(face); // e.g. (-1,0,0)
          CellIterator f_iter = 
            finePatch->getFaceCellIterator(face,"alongInteriorFaceCells");
          cerr << "C/F Face " << face << " offset " << offset << "\n";
          const int numStencilEntries = 1;
          int stencilEntries[numStencilEntries] = {face};
          for(; !f_iter.done(); f_iter++) {
            // For each fine cell at C/F interface, compute the index of
            // the neighboring coarse cell (add offset = outward normal to
            // the face and divide by the refinement ratio to obtain
            // "coarseLevel"-level index. Then add the connection between
            // the fine and coarse cells to the graph.
            IntVector fineCell = *f_iter;                        // in patch
            IntVector coarseCell = (fineCell + offset) / refRat; // outside p.
            const int numGraphEntries = 1;
            int graphEntries[numGraphEntries] = {entry[coarseCell]};
            entry[coarseCell]++;    // Update #unstructured connection counter
            cerr << "C-F matrix" 
                 << " cLev " << coarseLevel << " cCell " << coarseCell
                 << " (fLev " << fineLevel << " fCell " << fineCell << ")"
                 << " entry " << graphEntries[0];
            const double* graphValues = &AF[fineCell][face];
            cerr << " value " << graphValues[0];
            HYPRE_SStructMatrixSetValues(HA, coarseLevel,
                                         coarseCell.get_pointer(),
                                         CC_VAR, numGraphEntries, graphEntries,
                                         const_cast<double*>(graphValues));
            cerr << " done" << "\n";
            // The corresponding C-C structured connection is already set to 0
            // in impAMRICE.cc.
            // Hmmm, maybe not. Let's do it outselves.
            const double* stencilValues = &ZERO;
            cerr << "C-C " 
                 << " cLev " << _level << " cCell " << coarseCell
                 << " entry " << stencilEntries[0]
                 << " value " << stencilValues[0];
            HYPRE_SStructMatrixSetValues(HA, _level,
                                         coarseCell.get_pointer(),
                                         CC_VAR, numStencilEntries, stencilEntries,
                                         const_cast<double*>(stencilValues));
            cerr << " done" << "\n";
            
          }  // coarse cell interator
        }  // coarseFineInterface faces
      }  // patch has a coarseFineInterface
    }  // finePatch loop 
  } // end if viewpoint == DoingCoarseToFine
} // end HyprePatch_CC::makeConnections(matrix)

void
HypreDriverSStruct::HyprePatch_CC::getSolution
(HYPRE_SStructVector& HX,
 DataWarehouse* new_dw,
 const VarLabel* X_label,
 const bool modifies_x)
  //___________________________________________________________________
  // HypreDriverSStruct::HyprePatch_CC::makeInteriorVector~
  // Read the vector HV from Uintah into Hypre. HV can be the RHS
  // or the solution (initial guess) vector. HV is defined at the interior
  // cells of each patch.
  //___________________________________________________________________
{
  // Initialize pointers to data, cells
  typedef CCTypes::sol_type sol_type;
  sol_type Xnew;
  if (modifies_x) {
    new_dw->getModifiable(Xnew, X_label, _matl, _patch);
  } else {
    cerr << "  AllocateAndPut Xnew" 
         << " X_label " << X_label
         << " _matl " << _matl 
         << " _patch " << *_patch << "\n";
    new_dw->allocateAndPut(Xnew, X_label, _matl, _patch);
  }
#if 0
  // TESTING
  Xnew.initialize(0.0);
#else
  // Get the solution back from hypre. Note: because the data is
  // sorted in the same way in Uintah and Hypre, we can optimize by
  // read chunks of the vector rather than individual entries.
  for(int z = _low.z(); z <= _high.z(); z++) {
    for(int y = _low.y(); y <= _high.y(); y++) {
      const double* values = &Xnew[IntVector(_low.x(), y, z)];
      IntVector chunkLow(_low.x(), y, z);
      IntVector chunkHigh(_high.x(), y, z);
      // Feed data from Hypre to Uintah
      HYPRE_SStructVectorGetBoxValues(HX, _level,
                                      chunkLow.get_pointer(),
                                      chunkHigh.get_pointer(),
                                      CC_VAR, const_cast<double*>(values));
    }
  }
#endif
} // end HyprePatch_CC::makeInteriorVector()

//#####################################################################
// Utilities
//#####################################################################

void printLine(const string& s, const unsigned int len)
{
  for (unsigned int i = 0; i < len; i++) {
    cerr << s;
  }
  cerr << "\n";
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

#endif // #if HAVE_HYPRE_1_9
