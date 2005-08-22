/*--------------------------------------------------------------------------
 * File: HypreDriver.cc
 *
 * Implementation of a wrapper of a Hypre solver for a particular variable
 * type. 
 *--------------------------------------------------------------------------*/
// TODO: (taken from HypreSolver.cc)
// Matrix file - why are ghosts there?
// Read hypre options from input file
// 3D performance
// Logging?
// Report mflops
// Use a symmetric matrix whenever possible
// More efficient set?
// Reuse some data between solves?
// Where is the initial guess taken from and where to read & print it here?
//   (right now in initialize() and solve()).

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
  // TODO: Check if we need to subtract (1,1,1) from high or not.
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
  // Destroy matrix, RHS, solution objects
  cout_doing << "Destroying SStruct matrix, RHS, solution objects" << "\n";
  HYPRE_SStructMatrixDestroy(_HA);
  HYPRE_SStructVectorDestroy(_HB);
  HYPRE_SStructVectorDestroy(_HX);
  
  // Destroy graph objects
  cout_doing << "Destroying Solver object" << "\n";
  HYPRE_SStructGraphDestroy(_graph);
  
  // Destroying grid, stencil
  HYPRE_SStructStencilDestroy(_stencil);
  delete _vars;
  HYPRE_SStructGridDestroy(_grid);
}

void
HypreDriverSStruct::printMatrix(const string& fileName /* =  "output" */)
{
  cout_doing << "HypreDriverSStruct::printMatrix() begin" << "\n";
  if (!_params->printSystem) return;
  HYPRE_SStructMatrixPrint((fileName + ".sstruct").c_str(), _HA, 0);
  if (_requiresPar) {
    HYPRE_ParCSRMatrixPrint(_HA_Par, (fileName + ".par").c_str());
    // Print CSR matrix in IJ format, base 1 for rows and cols
    HYPRE_ParCSRMatrixPrintIJ(_HA_Par, 1, 1, (fileName + ".ij").c_str());
  }
  cout_doing << "HypreDriverSStruct::printMatrix() end" << "\n";
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
  typedef CCTypes::sol_type sol_type;
  ASSERTEQ(sizeof(Stencil7), 7*sizeof(double));

  //==================================================================
  // Set up the grid
  //==================================================================
  /* Create an empty grid in 3 dimensions with # parts = numLevels. */
  const int numDims = 3;
  const int numLevels = _level->getGrid()->numLevels();
  HYPRE_SStructGridCreate(_pg->getComm(), numDims, numLevels, &_grid);
  _vars = new HYPRE_SStructVariable[CC_NUM_VARS];
  _vars[CC_VAR] = HYPRE_SSTRUCT_VARIABLE_CELL; // We use only cell centered var

  // Loop over the Uintah patches that this proc owns
  for (int p = 0 ; p < _patches->size(); p++) {
    HyprePatch_CC hpatch(_patches->get(p),matl); // Read Uintah patch data
    hpatch.addToGrid(_grid,_vars);
  }
  HYPRE_SStructGridAssemble(_grid);

  //==================================================================
  // Set up the stencil
  //==================================================================
  // Prepare index offsets and stencil size
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

  //==================================================================
  // Set up the SStruct unstructured connection graph _graph
  //==================================================================
  // Create an empty graph
  HYPRE_SStructGraphCreate(_pg->getComm(), _grid, &_graph);
  // For ParCSR-requiring solvers like AMG
  if (_requiresPar) {
    cout_doing << "graph object type set to HYPRE_PARCSR" << "\n";
    HYPRE_SStructGraphSetObjectType(_graph, HYPRE_PARCSR);
  }

  //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
  //  Add structured equations (stencil-based) at the interior of
  //  each patch at every level to the graph.
  //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
  printLine("*",50);
  cout_doing << "Graph structured (interior) connections" << "\n";
  printLine("*",50);
  for (int level = 0; level < numLevels; level++) {
    cout_doing << "  Initializing graph stencil at level " << level
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
  cout_doing << "Graph unstructured (C/F) connections" << "\n";
  printLine("*",50);

  // Add Uintah patches that this proc owns
  for (int p = 0; p < _patches->size(); p++) {
    // Read Uintah patch info into our data structure
    const Patch* patch = _patches->get(p);
    HyprePatch_CC hpatch(_patches->get(p),matl); // Read Uintah patch data
    int level = hpatch.getLevel();
    printLine("$",40);
    cout_dbg << "Processing Patch" << "\n" << hpatch << "\n";
    printLine("$",40);
    
    if ((level > 0) && (patch->hasCoarseFineInterfaceFace())) {
      // If not at coarsest level, add fine-to-coarse connections at all
      // C/F interface faces.
      printLine("=",50);
      cout_doing << "Building fine-to-coarse connections" << "\n";
      printLine("=",50);
      patch->printPatchBCs(cout_dbg);
      hpatch.makeConnections(_graph,DoingFineToCoarse);
    } // end if (level > 0) and (patch has a CFI)

    if (level < numLevels-1) {
      // If not at finest level, examine the connection list that
      // impAMRICE.cc provides us and add the coarse-to-fine
      // connections to the Hypre graph. This list should cover all
      // the C/F interfaces of all next-finer level patches inscribed
      // in this patch.
      printLine("=",50);
      cout_doing << "Building coarse-to-fine connections" << "\n";
      printLine("=",50);
      hpatch.makeConnections(_graph,DoingCoarseToFine);
    } // end if (level < numLevels-1)
  } // end for p (patches)
  
  // Assemble the graph
  HYPRE_SStructGraphAssemble(_graph);
  cout_doing << "Assembled graph, nUVentries = "
             << hypre_SStructGraphNUVEntries(_graph) << "\n";

  //==================================================================
  // Set up the Struct left-hand-side matrix _HA
  //==================================================================
  // Create and initialize an empty SStruct matrix
  HYPRE_SStructMatrixCreate(_pg->getComm(), _graph, &_HA);

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

  //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
  // Add structured equation (stencil-based) entries at the interior of
  // each patch at every level to the matrix.
  //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
  printLine("*",50);
  cout_doing << "Matrix structured (interior) entries" << "\n";
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
  cout_doing << "Matrix unstructured (C/F) entries" << "\n";
  printLine("*",50);

  // Add Uintah patches that this proc owns
  for (int p = 0; p < _patches->size(); p++) {
    // Read Uintah patch info into our data structure
    const Patch* patch = _patches->get(p);
    HyprePatch_CC hpatch(_patches->get(p),matl); // Read Uintah patch data
    int level = hpatch.getLevel();
    printLine("$",40);
    cout_dbg << "Processing Patch" << "\n" << hpatch << "\n";
    printLine("$",40);
    
    if ((level > 0) && (patch->hasCoarseFineInterfaceFace())) {
      // If not at coarsest level, add fine-to-coarse connections at all
      // C/F interface faces.
      printLine("=",50);
      cout_doing << "Building fine-to-coarse entries" << "\n";
      printLine("=",50);
      patch->printPatchBCs(cout_dbg);
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
      cout_doing << "Building coarse-to-fine entries" << "\n";
      printLine("=",50);
      hpatch.makeConnections(_HA, _A_dw, _A_label,
                             _stencilSize, DoingCoarseToFine);
    } // end if (level < numLevels-1)
  } // end for p (patches)

  // This is an all-proc collective call
  HYPRE_SStructMatrixAssemble(_HA);

  //==================================================================
  // Set up the Struct right-hand-side vector _HB
  //==================================================================
  HYPRE_SStructVectorCreate(_pg->getComm(), _grid, &_HB);
  // For solvers that require ParCSR format
  if (_requiresPar) {
    HYPRE_SStructVectorSetObjectType(_HB, HYPRE_PARCSR);
  }
  HYPRE_SStructVectorInitialize(_HB);

  // Set RHS vector entries at the interior of
  // each patch at every level to the matrix.
  printLine("*",50);
  cout_doing << "Set RHS vector entries" << "\n";
  printLine("*",50);
  for (int p = 0 ; p < _patches->size(); p++) {
    // Read Uintah patch info into our data structure, set Uintah pointers
    const Patch* patch = _patches->get(p);
    HyprePatch_CC hpatch(patch,matl); // Read Uintah patch data
    hpatch.makeInteriorVector(_HB, _b_dw, _B_label);
  } // end for p (patches)

  HYPRE_SStructVectorAssemble(_HB);

  //==================================================================
  // Set up the Struct solution vector _HX
  //==================================================================
  HYPRE_SStructVectorCreate(_pg->getComm(), _grid, &_HX);
  // For solvers that require ParCSR format
  if (_requiresPar) {
    HYPRE_SStructVectorSetObjectType(_HX, HYPRE_PARCSR);
  }
  HYPRE_SStructVectorInitialize(_HX);

  // Set solution (initial guess) vector entries at the interior of
  // each patch at every level to the matrix.
  printLine("*",50);
  cout_doing << "Set solution vector initial guess entries" << "\n";
  printLine("*",50);
  for (int p = 0 ; p < _patches->size(); p++) {
    // Read Uintah patch info into our data structure, set Uintah pointers
    const Patch* patch = _patches->get(p);
    HyprePatch_CC hpatch(patch,matl); // Read Uintah patch data
    hpatch.makeInteriorVector(_HX, _guess_dw, _guess_label);
  } // end for p (patches)

  HYPRE_SStructVectorAssemble(_HX);

  // For solvers that require ParCSR format
  if (_requiresPar) {
    HYPRE_SStructMatrixGetObject(_HA, (void **) &_HA_Par);
    HYPRE_SStructVectorGetObject(_HB, (void **) &_HB_Par);
    HYPRE_SStructVectorGetObject(_HX, (void **) &_HX_Par);
  }
} // end HypreDriverSStruct::makeLinearSystem_CC()


void
HypreDriverSStruct::getSolution_CC(const int matl)
  //_____________________________________________________________________
  // Function HypreDriverSStruct::getSolution_CC~
  // Get the solution vector for a multi-level, CC variable problem from
  // the Hypre SStruct interface.
  //_____________________________________________________________________*/
{
#if 0
  typedef CCTypes::sol_type sol_type;
  // Loop over the Uintah patches that this proc owns
  for (int p = 0 ; p < _patches->size(); p++) {
    //==================================================================
    // Find patch extents and level it belongs to
    //==================================================================
    const Patch* patch = _patches->get(p);
    const int level = patch->getLevel()->getIndex();
    IntVector low  = patch->getInteriorCellLowIndex();
    IntVector high = patch->getInteriorCellHighIndex()-IntVector(1,1,1);
    // TODO: Check if we need to subtract (1,1,1) from high or not.
    
    //==================================================================
    // Read data from Hypre into Uintah
    //==================================================================
    // Initialize pointers to data, cells
    CellIterator iter(l, h);
    sol_type Xnew;
    if (_modifies_x) {
      _new_dw->getModifiable(Xnew, _X_label, matl, patch);
    } else {
      _new_dw->allocateAndPut(Xnew, _X_label, matl, patch);
    }
    // Get the solution back from hypre. Note: because the data is
    // sorted in the same way in Uintah and Hypre, we can optimize by
    // read chunks of the vector rather than individual entries.
    for (int z = low.z(); z < high.z(); z++) {
      for (int y = low.y(); y < high.y(); y++) {
        // This chunk of Hypre data has fixed y- and z- indices, and
        // running x-index.
        const double* values = &Xnew[IntVector(l.x(), y, z)];
        IntVector chunkLow(low.x(), y, z);
        IntVector chunkHigh(high.x()-1, y, z); // TODO: need the -1 ???
        HYPRE_SStructVectorGetBoxValues(_HX, level,
                                        chunkLow.get_pointer(),
                                        chunkHigh.get_pointer(),
                                        CC_VAR, const_cast<double*>(values));
      }
    }
  }
#endif
} // end HypreDriverSStruct::getSolution_CC()

void
HypreDriverSStruct::HyprePatch_CC::addToGrid
(HYPRE_SStructGrid& grid,
 HYPRE_SStructVariable* vars)
  //___________________________________________________________________
  // Add this patch to the Hypre grid
  //___________________________________________________________________
{
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
  if (viewpoint == DoingFineToCoarse) {
    //==================================================================
    // Add fine-to-coarse connections to graph
    //==================================================================
    const int fineLevel = _level;
    const int coarseLevel = _level-1;
    const IntVector& refRat = _patch->getLevel()->getRefinementRatio();
    vector<Patch::FaceType>::const_iterator iter;  
    // Loop over all C/F interface faces (note: a Uintah patch face is
    // marked as a coarsefine interface only when the current patch is
    // the FINE patch (what Todd calls "looking up is easy, not
    // looking down").  See also ICE/impAMRICE.cc,
    // ICE::matrixBC_CFI_finePatch().
   for (iter  = _patch->getCoarseFineInterfaceFaces()->begin(); 
         iter != _patch->getCoarseFineInterfaceFaces()->end(); ++iter) {
      Patch::FaceType face = *iter;                   // e.g. xminus=0
      IntVector offset = _patch->faceDirection(face); // e.g. (-1,0,0)
      CellIterator f_iter = 
        _patch->getFaceCellIterator(face,"alongInteriorFaceCells");
      for(; !f_iter.done(); f_iter++) {
        // For each fine cell at C/F interface, compute the index of
        // the neighboring coarse cell (add offset = outward normal to
        // the face and divide by the refinement ratio to obtain
        // "coarseLevel"-level index. Then add the connection between
        // the fine and coarse cells to the graph.
        IntVector fineCell = *f_iter;                        // inside patch
        IntVector coarseCell = (fineCell + offset) / refRat; // outside patch
        cout_doing << "Adding F->C connection to graph" << "\n";
        HYPRE_SStructGraphAddEntries(graph,
                                     fineLevel,fineCell.get_pointer(),
                                     CC_VAR,
                                     coarseLevel,coarseCell.get_pointer(),
                                     CC_VAR);
        cout_doing << "HYPRE call done" << "\n";
      }
    }
  } else { // now viewpoint == DoingCoarseToFine
    //==================================================================
    // Add coarse-to-fine connections to graph, requires ICE list of
    // C/F connections of this patch
    //==================================================================
    // TODO: this requires the list of connections to be prepared by ICE.
    // Add this code later.
  }
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
  CCTypes::matrix_type A;
  A_dw->get(A, A_label, _matl, _patch, Ghost::None, 0);
  if (symmetric) {
    //==================================================================
    // Add symmetric stencil equations to HA
    //==================================================================
    // Because AA is 7-point and the stencil is 4-point, copy data from AA
    // into stencil, and then feed it to Hypre
    double* values = new double[(_high.x()-_low.x())*stencilSize];
    int stencil_indices[] = {0,1,2,3};
    for(int z = _low.z(); z < _high.z(); z++) {
      for(int y = _low.y(); y < _high.y(); y++) {
        // Read data in "chunks" of fixed y-, z- index and running x-index
        const Stencil7* AA = &A[IntVector(_low.x(), y, z)];
        double* p = values;
        for (int x = _low.x(); x < _high.x(); x++) {
          *p++ = AA->p;
          *p++ = AA->w;
          *p++ = AA->s;
          *p++ = AA->b;
          AA++;
        }
        IntVector chunkLow(_low.x(), y, z);
        IntVector chunkHigh(_high.x()-1, y, z);
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
    // AA is 7-point and stencil is 7-point, feed data directly to Hypre
    int stencil_indices[] = {0,1,2,3,4,5,6};
    for(int z = _low.z(); z < _high.z(); z++) {
      for(int y = _low.y(); y < _high.y(); y++) {
        const double* values = &A[IntVector(_low.x(), y, z)].p;
        IntVector chunkLow(_low.x(), y, z);
        IntVector chunkHigh(_high.x()-1, y, z);
        // Feed data from Uintah to Hypre
        HYPRE_SStructMatrixSetBoxValues(HA, _level,
                                        chunkLow.get_pointer(),
                                        chunkHigh.get_pointer(), CC_VAR,
                                        stencilSize, stencil_indices,
                                        const_cast<double*>(values));
      }
    }
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
  for(int z = _low.z(); z < _high.z(); z++) {
    for(int y = _low.y(); y < _high.y(); y++) {
      const double* values = &V[IntVector(_low.x(), y, z)];
      IntVector chunkLow(_low.x(), y, z);
      IntVector chunkHigh(_high.x()-1, y, z);
      // Feed data from Uintah to Hypre
      HYPRE_SStructVectorSetBoxValues(HV, _level,
                                      chunkLow.get_pointer(),
                                      chunkHigh.get_pointer(), CC_VAR,
                                        const_cast<double*>(values));
    }
  }
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
  CCTypes::matrix_type A;
  A_dw->get(A, A_label, _matl, _patch, Ghost::None, 0);
  if (viewpoint == DoingFineToCoarse) {
    //==================================================================
    // Add fine-to-coarse entries to matrix
    //==================================================================
    //    const int fineLevel = _level;
    //    const int coarseLevel = _level-1;
    const IntVector& refRat = _patch->getLevel()->getRefinementRatio();
    vector<Patch::FaceType>::const_iterator iter;  
    // Loop over all C/F interface faces (note: a Uintah patch face is
    // marked as a coarsefine interface only when the current patch is
    // the FINE patch (what Todd calls "looking up is easy, not
    // looking down").  See also ICE/impAMRICE.cc,
    // ICE::matrixBC_CFI_finePatch().
   for (iter  = _patch->getCoarseFineInterfaceFaces()->begin(); 
         iter != _patch->getCoarseFineInterfaceFaces()->end(); ++iter) {
      Patch::FaceType face = *iter;                   // e.g. xminus=0
      IntVector offset = _patch->faceDirection(face); // e.g. (-1,0,0)
      CellIterator f_iter =
        _patch->getFaceCellIterator(face,"alongInteriorFaceCells");
      // Add one entry to one equation at a time: the equation at fineCell,
      // and the entry no. stencilSize (0..stencilSize-1 are the structured
      // entries, with entry "face" set to 0 inside ICE, and replaced with
      // the following unstructured (graph) entry.
      const int numEntries = 1;
      int stencilEntries[numEntries] = {stencilSize};
      for(; !f_iter.done(); f_iter++) {
        // For each fine cell at C/F interface, compute the index of
        // the neighboring coarse cell (add offset = outward normal to
        // the face and divide by the refinement ratio to obtain
        // "coarseLevel"-level index. Then add the connection between
        // the fine and coarse cells to the graph.
        IntVector fineCell = *f_iter;                        // inside patch
        IntVector coarseCell = (fineCell + offset) / refRat; // outside patch
        cout_doing << "Adding F->C entry to matrix" << "\n";
        const double* values = &A[fineCell][face];
        HYPRE_SStructMatrixSetBoxValues(HA, _level,
                                        fineCell.get_pointer(),
                                        fineCell.get_pointer(),
                                        CC_VAR, numEntries, stencilEntries,
                                        const_cast<double*>(values));
        cout_doing << "HYPRE call done" << "\n";
      }
    }
  } else { // now viewpoint == DoingCoarseToFine
    //==================================================================
    // Add coarse-to-fine entries to matrix, requires ICE list of
    // C/F connections of this patch
    //==================================================================
    // TODO: this requires the list of connections to be prepared by ICE.
    // Add this code later.
  }
} // end HyprePatch_CC::makeConnections(matrix)

//##############################################################
// OLD CODE FROM STAND ALONE. TODO: MOVE THIS CODE TO MAKELINEARSYSTEM_CC
// AND GETSOLUTION_CC.
//##############################################################

/*================= GRAPH CONSTRUCTION FUNCTIONS =========================*/

#if 0
void
HypreDriverSStruct::makeConnections(const ConstructionStatus& status,
                                    const CoarseFineViewpoint& viewpoint
                                    const HyprePatch& hpatch)
  // Build the C/F connections at the (d,s) C/F face of patch "patch"
  // at level "level" (connecting to level-1). We add fine-to-coarse
  // connections if viewpoint = FineToCoarse, otherwise we add
  // coarse-to-fine connections.
{
  printLine("=",50);
  cout_doing << "Building connections" 
             << "  level = " << level 
             << "  patch =\n" << *patch << "\n"
             << "  status = " << status
             << "  viewpoint = " << viewpoint << "\n"
             << "  Face d = " << d
             << " , s = " << s << "\n";
  printLine("=",50);
  const Counter numDims = _param->numDims;

  // Level info initialization
  Counter fineLevel, coarseLevel;
  if (viewpoint == FineToCoarse) {
    fineLevel = level;
    coarseLevel = level-1;
  } else { // viewpoint == CoarseToFine
    coarseLevel = level;
    fineLevel = level+1;
  }

  const Level* lev = hier._levels[fineLevel];
  const Vector<Counter>& refRat = lev->_refRat;
  const Vector<double>& h = lev->_meshSize;
  Vector<double> fineOffset = 0.5 * h;
  double cellVolume = h.prod();
  double faceArea = cellVolume / h[d];

  const Level* coarseLev = hier._levels[coarseLevel];
  const Vector<double>& coarseH = coarseLev->_meshSize;
  Vector<double> coarseOffset = 0.5 * coarseH;
  double coarseCellVolume = coarseH.prod();
  double coarseFaceArea = coarseCellVolume / coarseH[d];

  // Stencil info initialization
  Counter stencilSize = hypre_SStructStencilSize(stencil);

  // Fine cells of this face
  Box faceFineBox = patch->_box.faceExtents(d,s);
  // Coarse cells on the other side of the C/F boundary
  Box faceCoarseBox = faceFineBox.coarseNbhrExtents(refRat,d,s);

  cout_doing << "coarseLevel = " << coarseLevel << "\n";
  cout_doing << "fineLevel   = " << fineLevel   << "\n";
  // Loop over the C/F coarse cells and add connections
  for(Box::iterator coarse_iter = faceCoarseBox.begin();
      coarse_iter != faceCoarseBox.end(); ++coarse_iter) {
    // Compute the part fineCellFace of the fine face that directly
    // borders this coarse cell.
    Vector<int> cellFaceLower;
    if (s == LeftSide) {
      Vector<int> coarseCellOverFineCells = *coarse_iter;
      coarseCellOverFineCells[d] -= s;
      cellFaceLower = coarseCellOverFineCells * refRat;
    } else { // s == RightSide
      Vector<int> coarseCellOverFineCells = *coarse_iter;
      cellFaceLower = coarseCellOverFineCells * refRat;
      cellFaceLower[d] -= s;
    }
    Vector<int> offset = refRat - 1;
    offset[d] = 0;
    Vector<int> cellFaceUpper = cellFaceLower + offset;
    Box fineCellFace(cellFaceLower,cellFaceUpper);

    // Loop over the fine cells in fineCellFace and add their
    // connections to graph/matrix
    bool removeCCconnection = true; // C-C connection is removed once
    // per the loop over the fine cells below
    Counter fineCell = 0;
    for (Box::iterator fine_iter = fineCellFace.begin();
         fine_iter != fineCellFace.end(); ++fine_iter, ++fineCell) {
      cout_doing << "Coarse cell: " << *coarse_iter << "\n";
      cout_doing << "Fine   cell: " << *fine_iter   << "\n";

      if (status == Matrix) {
        //########################################################
        // Compute C-F interface flux.
        // Compute coordinates of:
        // This cell's center: xCell
        // The neighboring's cell data point: xNbhr
        // The face crossing between xCell and xNbhr: xFace
        //########################################################
        // xCell at this level, xNbhr at coarser level
        Vector<double> xCell = fineOffset + (h * (*fine_iter));
        Vector<double> xNbhr = coarseOffset + (coarseH * (*coarse_iter));
        // xFace is a convex combination of xCell, xNbhr with
        // weights depending on the distances of the coarse and fine
        // meshsizes, i.e., their distances in the d dimension from
        // the C/F boundary.
        double alpha = coarseH[d] / (coarseH[d] + h[d]);
        cout_doing << "alpha = " << alpha << "\n";
        Vector<double> xFace(0,numDims);
        for (Counter dd = 0; dd < numDims; dd++) {
          xFace[dd] = alpha*xCell[dd] + (1-alpha)*xNbhr[dd];
        }
        cout_doing << "xCell = " << xCell
                   << ", xFace = " << xFace
                   << ", xNbhr = " << xNbhr << "\n";
        
        //########################################################
        // Compute the harmonic average of the diffusion coefficient
        //########################################################
        double a    = 1.0; // Assumed constant a for now
        double diff = 0;
        for (Counter dd = 0; dd < numDims; dd++) {
          diff += pow(xNbhr[dd] - xCell[dd],2);
        }
        diff = sqrt(diff);
        double flux = a * faceArea / diff;
        cout_doing << "C/F flux = " << flux
                   << "   a = " << a
                   << " face = " << faceArea
                   << " diff = " << diff << "\n";
        
        if (viewpoint == FineToCoarse) {
          //====================================================
          // Compute matrix entries of the F->C graph connections
          //====================================================
          cout_doing << "Adding F->C flux to matrix" << "\n";
                    
          //########################################################
          // Add the flux to the fine cell equation - stencil part
          //########################################################
          const int numStencilEntries = 1;
          int stencilEntries[numStencilEntries] = {0};
          double stencilValues[numStencilEntries] = {flux};
          cout_doing << "Calling HYPRE_SStructMatrixAddToValues A (stencil)" << "\n";
          HYPRE_SStructMatrixAddToValues(_A,
                                         fineLevel, (*fine_iter).getData(),
                                         CC_VAR, numStencilEntries,
                                         stencilEntries,
                                         stencilValues);

          //########################################################
          // Add the C-F flux to the fine cell equation - graph part
          //########################################################
          const int numGraphEntries = 1;
          Counter entry = stencilSize;
          // Find the correct graph entry corresponding to this coarse
          // nbhr (subCoarse) of the fine cell subFine
          for (Counter dd = 0; dd < d; dd++) {
            // Are we near the ss-side boundary and is it a C/F bdry?
            for (BoxSide ss = LeftSide; ss <= RightSide; ++ss) {
              if ((patch->getBoundaryType(dd,ss) == Patch::CoarseFine) &&
                  ((*fine_iter)[dd] == patch->_box.get(ss)[dd])) {
                entry++;
              }
            }
          }
          if ((s == RightSide) &&
              (patch->getBoundaryType(d,s) == Patch::CoarseFine) &&
              ((*fine_iter)[d] == patch->_box.get(LeftSide)[d])) {
            entry++;
          }
          cout_doing << "entry (Fine cell -> coarse) = " << entry << "\n";
          int graphEntries[numGraphEntries] = {entry};
          double graphValues[numGraphEntries] = {-flux};
          cout_doing << "Calling HYPRE_SStructMatrixAddToValues A (graph)" << "\n";
          HYPRE_SStructMatrixAddToValues(_A,
                                         fineLevel,(*fine_iter).getData(),
                                         CC_VAR,numGraphEntries,
                                         graphEntries,
                                         graphValues);
        } else { // viewpoint == CoarseToFine
          //====================================================
          // Compute matrix entries of the C->F graph connections
          //====================================================
          cout_doing << "Adding C->F flux to matrix" << "\n";

          //########################################################
          // Add the C/F flux coarse cell equation - stencil part
          //########################################################
          const int numCoarseStencilEntries = 1;
          int coarseStencilEntries[numCoarseStencilEntries] = {0};
          double coarseStencilValues[numCoarseStencilEntries] = {flux};
          cout_doing << "Calling HYPRE_SStructMatrixAddToValues A (stencil)" << "\n";
          HYPRE_SStructMatrixAddToValues(_A,
                                         coarseLevel,(*coarse_iter).getData()
                                         ,CC_VAR,numCoarseStencilEntries,
                                         coarseStencilEntries,
                                         coarseStencilValues);

          //########################################################
          // Add the C/F flux coarse cell equation - graph part
          //########################################################
          const int numCoarseGraphEntries = 1;
          int coarseGraphEntries[numCoarseGraphEntries] =
            {stencilSize+fineCell};
          cout_doing << "fineCell = " << fineCell << "\n";
          cout_doing << "entry (coarse cell -> fine cell) = "
                     << stencilSize+fineCell << "\n";
          double coarseGraphValues[numCoarseGraphEntries] = {-flux};
          cout_doing << "Calling HYPRE_SStructMatrixAddToValues A (graph)" << "\n";
          HYPRE_SStructMatrixAddToValues(_A,
                                         coarseLevel, (*coarse_iter).getData(), CC_VAR,
                                         numCoarseGraphEntries,
                                         coarseGraphEntries,
                                         coarseGraphValues);
          if (removeCCconnection) {
            //########################################################
            // Remove the flux from the coarse nbhr to its coarse
            // stencil nbhr that underlies the fine patch.
            //########################################################
            // Compute coordinates of:
            // This cell's center: xCell
            // The neighboring's cell data point: xNbhr
            // The face crossing between xCell and xNbhr: xFace
            removeCCconnection = false;
            cout_doing << "Removing C/C flux connections from matrix" << "\n";
            BoxSide s2 = Side(-s);
            cout_doing << "      s = " << s
                       << ", s2 = " << s2 << "\n";
            xCell = coarseOffset + (coarseH * (*coarse_iter));
            xNbhr    = xCell;
            xNbhr[d] += s2*coarseH[d];
            xFace    = xCell;
            xFace[d] = 0.5*(xCell[d] + xNbhr[d]);
            cout_doing << "xCell = " << xCell
                       << ", xFace = " << xFace
                       << ", xNbhr = " << xNbhr << "\n";
            
            // Compute the harmonic average of the diffusion
            // coefficient
            double a    = 1.0; // Assumed constant a for now
            double diff = fabs(xNbhr[d] - xCell[d]);
            double flux = a * coarseFaceArea / diff;
            cout_doing << "      C/F flux = " << flux
                       << "   a = " << a
                       << " face = " << faceArea
                       << " diff = " << diff << "\n";
            const int coarseNumStencilEntries = 2;
            int coarseStencilEntries[coarseNumStencilEntries] =
              {0, 2*d + ((-s+1)/2) + 1};
            double coarseStencilValues[coarseNumStencilEntries] =
              {-flux, flux};
            HYPRE_SStructMatrixAddToValues(_A,
                                           coarseLevel,(*coarse_iter).getData(), CC_VAR,
                                           coarseNumStencilEntries,
                                           coarseStencilEntries,
                                           coarseStencilValues);
          } // end if (fineCell == 0)
        } // end if viewpoint
      } // end if status
    } // end for fine_iter
  } // end for coarse_iter
} // end makeConnections
#endif

#if 0
void
HypreDriverSStruct::makeLinearSystem(const Hierarchy& hier,
                                     const HYPRE_SStructGrid& grid,
                                     const HYPRE_SStructStencil& stencil)
  //_____________________________________________________________________
  // Function HypreDriverSStruct::makeLinearSystem~
  // Initialize the linear system: set up the values on the links of the
  // graph of the LHS matrix A and value of the RHS vector b at all
  // patches of all levels. Delete coarse data underlying fine patches.
  //_____________________________________________________________________
{
  // Add interior equations

  //======================================================================
  // Add to graph the unstructured part of the stencil connecting the
  // coarse and fine level at every C/F boundary (F->C connections at
  // this patch's outer boundaries, and C->F connections at all
  // applicable C/F boundaries of next-finer-level patches that lie
  // above this patch.
  //======================================================================
  printLine("*",50);
  cout_doing << "Matrix unstructured (C/F) equations" << "\n";
  printLine("*",50);
  for (Counter level = 0; level < numLevels; level++) {
    const Level* lev = hier._levels[level];
    // Loop over patches of this proc
    for (Counter i = 0; i < lev->_patchList[MYID].size(); i++) {
      Patch* patch = lev->_patchList[MYID][i];
      printLine("%",40);
      cout_doing << "Processing Patch" << "\n"
                 << *patch << "\n";
      printLine("%",40);
      
      if (level > 0) {
        // If not at coarsest level,
        // loop over outer boundaries of this patch and add
        // fine-to-coarse connections
        printLine("=",50);
        cout_doing << "Building fine-to-coarse connections" << "\n";
        printLine("=",50);
        for (Counter d = 0; d < numDims; d++) {
          for (BoxSide s = LeftSide; s <= RightSide; ++s) {
            if (patch->getBoundaryType(d,s) == Patch::CoarseFine) {
              makeConnections(Matrix,hier,stencil,level,patch,d,s,FineToCoarse);
            } // end if boundary is CF interface
          } // end for s
        } // end for d
      }

      if (level < numLevels-1) {
        printLine("=",50);
        cout_doing << "Building coarse-to-fine connections" 
                   << " Patch ID = " << patch->_patchID
                   << "\n";
        printLine("=",50);
        //  const int numDims   = _param->numDims;
        const Vector<Counter>& refRat = hier._levels[level+1]->_refRat;
        // List of fine patches covering this patch
        vector<Patch*> finePatchList = hier.finePatchesOverMe(*patch);
        Box coarseRefined(patch->_box.get(LeftSide) * refRat,
                          (patch->_box.get(RightSide) + 1) * refRat - 1);
        cout_doing << "coarseRefined " << coarseRefined << "\n";
    
        //===================================================================
        // Loop over next-finer level patches that cover this patch
        //===================================================================
        for (vector<Patch*>::iterator iter = finePatchList.begin();
             iter != finePatchList.end(); ++iter) {
          //===================================================================
          // Compute relevant boxes at coarse and fine levels
          //===================================================================
          Patch* finePatch = *iter;
          cout_doing << "Considering patch "
                     << "ID=" << setw(2) << left << finePatch->_patchID << " "
                     << "owner=" << setw(2) << left << finePatch->_procID << " "
                     << finePatch->_box << " ..." << "\n";
          // Intersection of fine and coarse patches in fine-level subscripts
          Box fineIntersect = coarseRefined.intersect(finePatch->_box);
          // Intersection of fine and coarse patches in coarse-level subscripts
          Box coarseUnderFine(fineIntersect.get(LeftSide) / refRat, 
                              fineIntersect.get(RightSide) / refRat);
          
          //===================================================================
          // Loop over coarse-to-fine internal boundaries of the fine patch;
          // add C-to-F connections; delete the old C-to-coarseUnderFine
          // connections.
          //===================================================================
          for (Counter d = 0; d < numDims; d++) {
            for (BoxSide s = LeftSide; s <= RightSide; ++s) {
              if (finePatch->getBoundaryType(d,s) == Patch::CoarseFine) {
                // Coarse cell nbhring the C/F boundary on the other side
                // of the fine patch
                Box faceCoarseBox =
                  fineIntersect.coarseNbhrExtents(refRat,d,s);
                if (patch->_box.intersect(faceCoarseBox).degenerate()) {
                  // The coarse nbhrs of this fine patch C/F boundary are
                  // outside this coarse patch, ignore this face.
                  continue;
                }
                // Set the coarse-to-fine connections
                makeConnections(Matrix,level,finePatch,d,s,CoarseToFine);

              } // end if CoarseFine boundary
            } // end for s
          } // end for d
        } // end for all fine patches that cover this patch
      } // end if (level < numLevels-1)
    } // end for i (patches)
  } // end for level
} // end makeLinearSystem()
#endif

//#####################################################################
// Utilities
//#####################################################################

void printLine(const string& s, const unsigned int len)
{
  for (unsigned int i = 0; i < len; i++) {
    cout_doing << s;
  }
  cout_doing << "\n";
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

#if 0
// TODO: move this function to impAMRICE !!!!!
double harmonicAvg(const Point& x,
                   const Point& y,
                   const Point& z,
                   const double& Ax,
                   const double& Ay)
  /*_____________________________________________________________________
    Function harmonicAvg~: 
    Harmonic average of the diffusion coefficient.
    A = harmonicAvg(X,Y,Z) returns the harmonic average of the
    diffusion coefficient a(T) (T in R^D) along the line connecting
    the points X,Y in R^D. That is, A = 1/(integral_0^1
    1/a(t1(s),...,tD(s)) ds), where td(s) = x{d} + s*(y{d} -
    x{d})/norm(y-x) is the arclength parameterization of the
    d-coordinate of the line x-y, d = 1...D.  We assume that A is
    piecewise constant with jump at Z (X,Y are normally cell centers
    and Z at the cell face). X,Y,Z are Dx1 location arrays.  In
    general, A can be analytically computed for the specific cases we
    consider; in general, use some simple quadrature formula for A
    from discrete a-values. This can be implemented by the derived
    test cases from Param.

    ### NOTE: ### If we use a different
    refinement ratio in different dimensions, near the interface we
    may need to compute A along lines X-Y that cross more than one
    cell boundary. This is currently ignored and we assume all lines
    cut one cell interface only.
    _____________________________________________________________________*/

{
  const int numDims = 3;
  /* Compute distances x-y and x-z */
  double dxy = 0.0, dxz = 0.0;
  for (int d = 0; d < numDims; d++) {
    dxy += pow(fabs(y(d) - x(d)),2.0);
    dxz += pow(fabs(z(d) - x(d)),2.0);
  }
  double K = sqrt(dxz/dxy);
  return (Ax*Ay)/((1-K)*Ax + K*Ay);
}
#endif
