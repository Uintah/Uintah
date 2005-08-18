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

namespace Uintah{

  void
  testing()
  {
    const Patch * patch;
    cout << *patch;
  }

}

//__________________________________
//  To turn on normal output
//  setenv SCI_DEBUG "HYPRE_DOING_COUT:+"

static DebugStream cout_doing("HYPRE_DOING_COUT", false);
static const int NUM_VARS = 1;
static void linePrint(const string& s, const unsigned int len);

//#####################################################################
// class HypreDriver implementation common to all variable types
//#####################################################################

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
  const Patch * asdfpatch;
  cout << *asdfpatch;

  typedef CCTypes::sol_type sol_type;
  ASSERTEQ(sizeof(Stencil7), 7*sizeof(double));

  //==================================================================
  // Set up the grid
  //==================================================================
  /* Create an empty grid in 3 dimensions with # parts = numLevels. */
  const int numDims = 3;
  const int numLevels = _level->getGrid()->numLevels();
  HYPRE_SStructGridCreate(_pg->getComm(), numDims, numLevels, &_grid);
  HYPRE_SStructVariable vars[NUM_VARS] =
    {HYPRE_SSTRUCT_VARIABLE_CELL}; // We use only cell centered vars

  // Add Uintah patches that this proc owns
  for (int p = 0 ; p < _patches->size(); p++) {
    // Find patch extents and level it belongs to
    const Patch* patch = _patches->get(p);
    const int level = patch->getLevel()->getIndex();
    IntVector low  = patch->getInteriorCellLowIndex();
    IntVector high = patch->getInteriorCellHighIndex()-IntVector(1,1,1);
    // TODO: Check if we need to subtract (1,1,1) from high or not.
    
    // Add this patch to the Hypre grid
    HYPRE_SStructGridSetExtents(_grid, level, low.get_pointer(), high.get_pointer());
    HYPRE_SStructGridSetVariables(_grid, level, NUM_VARS, vars);
  }
  HYPRE_SStructGridAssemble(_grid);

  //==================================================================
  // Set up the stencil
  //==================================================================
  if (_params->symmetric) {
    HYPRE_SStructStencilCreate(numDims, numDims+1, &_stencil);
    int offsets[numDims+1][numDims] = {{0,0,0},
                                       {-1,0,0},
                                       {0,-1,0},
                                       {0,0,-1}};
    for (int i = 0; i < 4; i++) {
      HYPRE_SStructStencilSetEntry(_stencil, i, offsets[i], 0);
    }
  } else {
    HYPRE_SStructStencilCreate(numDims, 2*numDims+1, &_stencil);
    int offsets[2*numDims+1][numDims] = {{0,0,0},
                                         {1,0,0}, {-1,0,0},
                                         {0,1,0}, {0,-1,0},
                                         {0,0,1}, {0,0,-1}};
    for (int i = 0; i < 7; i++) {
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

  //######################################################################
  //  Add structured equations (stencil-based) at the interior of
  //  each patch at every level to the graph.
  //######################################################################
  linePrint("*",50);
  cout_doing << "Graph structured (interior) connections" << "\n";
  linePrint("*",50);
  for (int level = 0; level < numLevels; level++) {
    cout_doing << "  Initializing graph stencil at level " << level
               << " of " << numLevels << "\n";
    HYPRE_SStructGraphSetStencil(_graph, level, 0, _stencil);
  }

  //######################################################################
  // Add to graph the unstructured part of the stencil connecting the
  // coarse and fine level at every C/F boundary (F->C connections at
  // this patch's outer boundaries, and C->F connections at all
  // applicable C/F boundaries of next-finer-level patches that lie
  // above this patch.
  //######################################################################
  linePrint("*",50);
  cout_doing << "Graph unstructured (C/F) connections" << "\n";
  linePrint("*",50);

  // Add Uintah patches that this proc owns
  for (int p = 0; p < _patches->size(); p++) {
    // Find patch extents and level it belongs to
    const Patch* patch = _patches->get(p);
    const int level = patch->getLevel()->getIndex();
    IntVector low  = patch->getInteriorCellLowIndex();
    IntVector high = patch->getInteriorCellHighIndex()-IntVector(1,1,1);
    // TODO: Check if we need to subtract (1,1,1) from high or not.

    Patch * thePatch;
    cout << *thePatch;

    linePrint("$",40);
    cout_doing << "Processing Patch" << "\n"
               << *patch << "\n";
    linePrint("$",40);
    
    if (level > 0) {
      // If not at coarsest level, loop over outer boundaries of this
      // patch and add fine-to-coarse connections
      linePrint("=",50);
      cout_doing << "Building fine-to-coarse connections" << "\n";
      linePrint("=",50);
      for (int d = 0; d < numDims; d++) {
        for (BoxSide s = LeftSide; s <= RightSide; ++s) {
          if (patch->getBoundaryType(d,s) == Patch::CoarseFine) {
            cout_doing << "boundary is " << patch->getBoundaryType(d,s) << "\n";
            makeConnections_FC(Graph,level,patch,d,s);
          } // end if boundary is CF interface
        } // end for s
      } // end for d
    }

    if (level < numLevels-1) {
      // If not at finest level, examine the connection list that
      // impAMRICE.cc provides us and add the coarse-to-fine
      // connections to the Hypre graph one by one (this is done
      // inside makeConnections()).
      linePrint("=",50);
      cout_doing << "Building coarse-to-fine connections" << "\n";
      linePrint("=",50);
      makeConnections_CF(Graph,level,patch);
      
    } // end if (level < numLevels-1)
  } // end for p (patches)
  
  // Assemble the graph
  HYPRE_SStructGraphAssemble(_graph);
  cout_doing << "Assembled graph, nUVentries = "
             << hypre_SStructGraphNUVEntries(_graph) << "\n";
  funcPrint("Solver::makeGraph()",FEnd);
} // end makeGraph()


  //==================================================================
  // Set up the Struct left-hand-side matrix _HA
  //==================================================================
HYPRE_StructMatrixCreate(_pg->getComm(), _grid, stencil, &_HA);
HYPRE_StructMatrixSetSymmetric(_HA, _params->symmetric);
int ghost[] = {1,1,1,1,1,1};
HYPRE_StructMatrixSetNumGhost(_HA, ghost);
// This works only for SStruct -> ParCSR. TODO: convert Struct -> SStruct
// -> ParCSR for complicated diffusion 1-level problems that need AMG.
//  if (_requiresPar) {
//    HYPRE_StructMatrixSetObjectType(_HA, HYPRE_PARCSR);
//  }
HYPRE_StructMatrixInitialize(_HA);

for(int p=0;p<_patches->size();p++){
  const Patch* patch = _patches->get(p);

  // Get the data from Uintah
  CCTypes::matrix_type A;
  _A_dw->get(A, _A_label, matl, patch, Ghost::None, 0);

  Patch::VariableBasis basis =
    Patch::translateTypeToBasis(sol_type::getTypeDescription()
                                ->getType(), true);
  IntVector ec = _params->getSolveOnExtraCells() ?
    IntVector(0,0,0) : -_level->getExtraCells();
  IntVector l = patch->getLowIndex(basis, ec);
  IntVector h = patch->getHighIndex(basis, ec);

  // Feed it to Hypre
  if(_params->symmetric){
    double* values = new double[(h.x()-l.x())*4];	
    int stencil_indices[] = {0,1,2,3};
    for(int z=l.z();z<h.z();z++){
      for(int y=l.y();y<h.y();y++){
        const Stencil7* AA = &A[IntVector(l.x(), y, z)];
        double* p = values;
        for(int x=l.x();x<h.x();x++){
          *p++ = AA->p;
          *p++ = AA->w;
          *p++ = AA->s;
          *p++ = AA->b;
          AA++;
        }
        IntVector ll(l.x(), y, z);
        IntVector hh(h.x()-1, y, z);
        HYPRE_StructMatrixSetBoxValues(_HA,
                                       ll.get_pointer(),
                                       hh.get_pointer(),
                                       4, stencil_indices, values);

      }
    }
    delete[] values;
  } else {
    int stencil_indices[] = {0,1,2,3,4,5,6};
    for(int z=l.z();z<h.z();z++){
      for(int y=l.y();y<h.y();y++){
        const double* values = &A[IntVector(l.x(), y, z)].p;
        IntVector ll(l.x(), y, z);
        IntVector hh(h.x()-1, y, z);
        HYPRE_StructMatrixSetBoxValues(_HA,
                                       ll.get_pointer(),
                                       hh.get_pointer(),
                                       7, stencil_indices,
                                       const_cast<double*>(values));
      }
    }
  }
}
HYPRE_StructMatrixAssemble(_HA);

//==================================================================
// Set up the Struct right-hand-side vector _HB
//==================================================================
HYPRE_StructVectorCreate(_pg->getComm(), _grid, &_HB);
// This works only for SStruct -> ParCSR. TODO: convert Struct -> SStruct
// -> ParCSR for complicated diffusion 1-level problems that need AMG.
//  if (_requiresPar) {
//    HYPRE_StructVectorSetObjectType(_HB, HYPRE_PARCSR);
//  }
HYPRE_StructVectorInitialize(_HB);

for(int p=0;p<_patches->size();p++){
  const Patch* patch = _patches->get(p);

  // Get the data from Uintah
  CCTypes::const_type B;
  _b_dw->get(B, _B_label, matl, patch, Ghost::None, 0);

  Patch::VariableBasis basis =
    Patch::translateTypeToBasis(sol_type::getTypeDescription()
                                ->getType(), true);
  IntVector ec = _params->getSolveOnExtraCells() ?
    IntVector(0,0,0) : -_level->getExtraCells();
  IntVector l = patch->getLowIndex(basis, ec);
  IntVector h = patch->getHighIndex(basis, ec);

  // Feed it to Hypre
  for(int z=l.z();z<h.z();z++){
    for(int y=l.y();y<h.y();y++){
      const double* values = &B[IntVector(l.x(), y, z)];
      IntVector ll(l.x(), y, z);
      IntVector hh(h.x()-1, y, z);
      HYPRE_StructVectorSetBoxValues(_HB,
                                     ll.get_pointer(),
                                     hh.get_pointer(),
                                     const_cast<double*>(values));
    }
  }
}
HYPRE_StructVectorAssemble(_HB);

//==================================================================
// Set up the Struct solution vector _HX
//==================================================================
HYPRE_StructVectorCreate(_pg->getComm(), _grid, &_HX);
// This works only for SStruct -> ParCSR. TODO: convert Struct -> SStruct
// -> ParCSR for complicated diffusion 1-level problems that need AMG.
//  if (_requiresPar) {
//    HYPRE_StructVectorSetObjectType(_HX, HYPRE_PARCSR);
//  }
HYPRE_StructVectorInitialize(_HX);

for(int p=0;p<_patches->size();p++){
  const Patch* patch = _patches->get(p);

  if (_guess_label) {
    CCTypes::const_type X;
    _guess_dw->get(X, _guess_label, matl, patch, Ghost::None, 0);

    // Get the initial guess from Uintah
    Patch::VariableBasis basis =
      Patch::translateTypeToBasis(sol_type::getTypeDescription()
                                  ->getType(), true);
    IntVector ec = _params->getSolveOnExtraCells() ?
      IntVector(0,0,0) : -_level->getExtraCells();
    IntVector l = patch->getLowIndex(basis, ec);
    IntVector h = patch->getHighIndex(basis, ec);

    // Feed it to Hypre
    for(int z=l.z();z<h.z();z++){
      for(int y=l.y();y<h.y();y++){
        const double* values = &X[IntVector(l.x(), y, z)];
        IntVector ll(l.x(), y, z);
        IntVector hh(h.x()-1, y, z);
        HYPRE_StructVectorSetBoxValues(_HX,
                                       ll.get_pointer(),
                                       hh.get_pointer(),
                                       const_cast<double*>(values));
      }
    }
  }  // initialGuess
} // patch loop
HYPRE_StructVectorAssemble(_HX);

// If solver requires ParCSR format, convert Struct to ParCSR.
// This works only for SStruct -> ParCSR. TODO: convert Struct -> SStruct
// -> ParCSR for complicated diffusion 1-level problems that need AMG.
//  if (_requiresPar) {
//    HYPRE_StructMatrixGetObject(_HA, (void **) &_HA_Par);
//    HYPRE_StructVectorGetObject(_HB, (void **) &_HB_Par);
//    HYPRE_StructVectorGetObject(_HX, (void **) &_HX_Par);
//  }

} // end HypreDriverSStruct::makeLinearSystem_CC()


void
HypreDriverSStruct::getSolution_CC(const int matl)
  //_____________________________________________________________________
  // Function HypreDriverSStruct::getSolution_CC~
  // Get the solution vector for a 1-level, CC variable problem from
  // the Hypre Struct interface.
  //_____________________________________________________________________*/
{
typedef CCTypes::sol_type sol_type;
for(int p=0;p<_patches->size();p++){
const Patch* patch = _patches->get(p);

Patch::VariableBasis basis =
Patch::translateTypeToBasis(sol_type::getTypeDescription()
  ->getType(), true);
IntVector ec = _params->getSolveOnExtraCells() ?
IntVector(0,0,0) : -_level->getExtraCells();
IntVector l = patch->getLowIndex(basis, ec);
IntVector h = patch->getHighIndex(basis, ec);
CellIterator iter(l, h);

sol_type Xnew;
if(_modifies_x)
  _new_dw->getModifiable(Xnew, _X_label, matl, patch);
else
_new_dw->allocateAndPut(Xnew, _X_label, matl, patch);
	
// Get the solution back from hypre
for(int z=l.z();z<h.z();z++){
for(int y=l.y();y<h.y();y++){
const double* values = &Xnew[IntVector(l.x(), y, z)];
IntVector ll(l.x(), y, z);
IntVector hh(h.x()-1, y, z);
HYPRE_StructVectorGetBoxValues(_HX,
  ll.get_pointer(),
  hh.get_pointer(),
  const_cast<double*>(values));
}
}
}
} // end HypreDriverSStruct::getSolution_CC()

//##############################################################
// OLD CODE FROM STAND ALONE. TODO: MOVE THIS CODE TO MAKELINEARSYSTEM_CC
// AND GETSOLUTION_CC.
//##############################################################
void
Solver::initializeSStruct(void)
{
  /*-----------------------------------------------------------
   * Set up the grid
   *-----------------------------------------------------------*/
  int time_index = hypre_InitializeTiming("SStruct Interface");
  hypre_BeginTiming(time_index);
  linePrint("-",50);
  cerr << "Set up the grid (AMR levels, patches)" << "\n";
  linePrint("-",50);
  hier.make();
  makeGrid(param, hier, grid);             // Make Hypre grid from hier
  //hier.printPatchBoundaries();
  cerr << "Printing hierarchy:" << "\n";
  cerr << hier;                            // Print the patch hierarchy

  /*-----------------------------------------------------------
   * Set up the stencils
   *-----------------------------------------------------------*/
  linePrint("-",50);
  cerr << "Set up the stencils on all the patchs" << "\n";
  linePrint("-",50);
  makeStencil(param, hier, stencil);

  makeGraph(hier, grid, stencil);
  initializeData(hier, grid);
  makeLinearSystem(hier, grid, stencil);
  assemble();

  /*-----------------------------------------------------------
   * Set up the SStruct matrix
   *-----------------------------------------------------------*/
  linePrint("-",50);
  cerr << "Initialize the solver (Set up SStruct graph, matrix)" << "\n";
  linePrint("-",50);
  solver->initialize(hier, grid, stencil);
    
  /* Print total time for setting up the grid, stencil, graph, solver */
  hypre_EndTiming(time_index);
  hypre_PrintTiming("SStruct Interface", MPI_COMM_WORLD);
  hypre_FinalizeTiming(time_index);
  hypre_ClearTiming();
    
  /*-----------------------------------------------------------
   * Print out the system and initial guess
   *-----------------------------------------------------------*/
  linePrint("-",50);
  cout_doing << "Print out the system and initial guess" << "\n";
  linePrint("-",50);
  solver->printMatrix("output_A");
  solver->printRHS("output_b");
  solver->printSolution("output_x0");



  funcPrint("Solver::initialize()",FEnd);
}

void
Solver::initializeData(const Hierarchy& hier,
                       const HYPRE_SStructGrid& grid)
{
  funcPrint("Solver::initializeData()",FBegin);

  // Create an empty matrix with the graph non-zero pattern
  HYPRE_SStructMatrixCreate(MPI_COMM_WORLD, _graph, &_A);
  cout_doing << "Created empty SStructMatrix" << "\n";
  // Initialize RHS vector b and solution vector x
  cout_doing << "Create empty b,x" << "\n";
  HYPRE_SStructVectorCreate(MPI_COMM_WORLD, grid, &_b);
  cout_doing << "Done b" << "\n";
  HYPRE_SStructVectorCreate(MPI_COMM_WORLD, grid, &_x);
  cout_doing << "Done x" << "\n";

  // If using AMG, set (A,b,x)'s object type to ParCSR now
  if (_requiresPar) {
    cout_doing << "Matrix object type set to HYPRE_PARCSR" << "\n";
    HYPRE_SStructMatrixSetObjectType(_A, HYPRE_PARCSR);
    cout_doing << "Vector object type set to HYPRE_PARCSR" << "\n";
    HYPRE_SStructVectorSetObjectType(_b, HYPRE_PARCSR);
    cout_doing << "Done b" << "\n";
    HYPRE_SStructVectorSetObjectType(_x, HYPRE_PARCSR);
    cout_doing << "Done x" << "\n";
  }
  cout_doing << "Init A" << "\n";
  HYPRE_SStructMatrixInitialize(_A);
  cout_doing << "Init b,x" << "\n";
  HYPRE_SStructVectorInitialize(_b);
  cout_doing << "Done b" << "\n";
  HYPRE_SStructVectorInitialize(_x);
  cout_doing << "Done x" << "\n";

  funcPrint("Solver::initializeData()",FEnd);
}

void
Solver::assemble(void)
{
  cout_doing << "Solver::assemble() end" << "\n";
  // Assemble the matrix - a collective call
  HYPRE_SStructMatrixAssemble(_A); 
  HYPRE_SStructVectorAssemble(_b);
  HYPRE_SStructVectorAssemble(_x);

  // For BoomerAMG solver: set up the linear system in ParCSR format
  if (_requiresPar) {
    HYPRE_SStructMatrixGetObject(_A, (void **) &_parA);
    HYPRE_SStructVectorGetObject(_b, (void **) &_parB);
    HYPRE_SStructVectorGetObject(_x, (void **) &_parX);
  }
  cout_doing << "Solver::assemble() begin" << "\n";
}



/*================= GRAPH CONSTRUCTION FUNCTIONS =========================*/
void
Solver::makeConnections(const ConstructionStatus& status,
                        const Hierarchy& hier,
                        const HYPRE_SStructStencil& stencil,
                        const Counter level,
                        const Patch* patch,
                        const Counter& d,
                        const Side& s,
                        const CoarseFineViewpoint& viewpoint)
  // Build the C/F connections at the (d,s) C/F face of patch "patch"
  // at level "level" (connecting to level-1). We add fine-to-coarse
  // connections if viewpoint = FineToCoarse, otherwise we add
  // coarse-to-fine connections.
{
  linePrint("=",50);
  cout_doing << "Building connections" 
             << "  level = " << level 
             << "  patch =\n" << *patch << "\n"
             << "  status = " << status
             << "  viewpoint = " << viewpoint << "\n"
             << "  Face d = " << d
             << " , s = " << s << "\n";
  linePrint("=",50);
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
    dbg.indent();
    bool removeCCconnection = true; // C-C connection is removed once
    // per the loop over the fine cells below
    Counter fineCell = 0;
    for (Box::iterator fine_iter = fineCellFace.begin();
         fine_iter != fineCellFace.end(); ++fine_iter, ++fineCell) {
      cout_doing << "Coarse cell: " << *coarse_iter << "\n";
      cout_doing << "Fine   cell: " << *fine_iter   << "\n";
      if (status == Graph) {
        if (viewpoint == FineToCoarse) {
          //====================================================
          // Add F->C connections to graph
          //====================================================
          cout_doing << "Adding F->C connection to graph" << "\n";
          HYPRE_SStructGraphAddEntries(_graph,
                                       fineLevel,(*fine_iter).getData(),0,
                                       coarseLevel,(*coarse_iter).getData(),0);
          cout_doing << "HYPRE call done" << "\n";
        } else { // viewpoint == CoarseToFine
          //====================================================
          // Add C->F connections to graph
          //====================================================
          cout_doing << "Adding C->F connection to graph" << "\n";
          HYPRE_SStructGraphAddEntries(_graph,
                                       coarseLevel,(*coarse_iter).getData(),0,
                                       fineLevel,(*fine_iter).getData(),0);
          cout_doing << "HYPRE call done" << "\n";
        } // end if viewpoint
      } else { // status == Matrix
        
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
                                         0, numStencilEntries,
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
                                         0,numGraphEntries,
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
                                         ,0,numCoarseStencilEntries,
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
                                         coarseLevel, (*coarse_iter).getData(), 0,
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
                                           coarseLevel,(*coarse_iter).getData(), 0,
                                           coarseNumStencilEntries,
                                           coarseStencilEntries,
                                           coarseStencilValues);
          } // end if (fineCell == 0)
        } // end if viewpoint
      } // end if status
    } // end for fine_iter
    dbg.unindent();
  } // end for coarse_iter
} // end makeConnections

void 
Solver::makeUnderlyingIdentity(const Counter level,
                               const HYPRE_SStructStencil& stencil,
                               const Box& coarseUnderFine)
  // Replace the matrix equations for the underlying coarse box
  // with the identity matrix.
{
  cout_doing << "Putting identity on underlying coarse data" << "\n"
             << "coarseUnderFine " << coarseUnderFine << "\n";
  Counter stencilSize = hypre_SStructStencilSize(stencil);
  int* entries = new int[stencilSize];
  for (Counter entry = 0; entry < stencilSize; entry++) {
    entries[entry] = entry;
  }
  const Counter numCoarseCells = coarseUnderFine.volume();
  double* values    = new double[stencilSize * numCoarseCells];
  double* rhsValues = new double[numCoarseCells];

  cout_doing << "Looping over cells in coarse underlying box:" 
             << "\n";
  Counter cell = 0;
  for (Box::iterator coarse_iter = coarseUnderFine.begin();
       coarse_iter != coarseUnderFine.end(); ++coarse_iter, cell++) {
    cout_doing << "cell = " << cell << " " << *coarse_iter << "\n";
    
    int offsetValues    = stencilSize * cell;
    int offsetRhsValues = cell;
    // Initialize the stencil values of this cell's equation to 0
    // except the central coefficient that is 1
    values[offsetValues] = 1.0;
    for (Counter entry = 1; entry < stencilSize; entry++) {
      values[offsetValues + entry] = 0.0;
    }
    // Set the corresponding RHS entry to 0.0
    rhsValues[offsetRhsValues] = 0.0;
  } // end for cell
  
  printValues(coarseUnderFine.volume(),stencilSize,values,rhsValues);
  
  // Effect the identity operator change in the Hypre structure
  // for A
  Box box(coarseUnderFine);
  cout_doing << "Calling HYPRE_SStructMatrixSetBoxValues A" << "\n";
  HYPRE_SStructMatrixSetBoxValues(_A, level,
                                  box.get(LeftSide).getData(),
                                  box.get(RightSide).getData(),
                                  0, stencilSize, entries, values);
  cout_doing << "Calling HYPRE_SStructVectorSetBoxValues b" << "\n";
  HYPRE_SStructVectorSetBoxValues(_b, level, 
                                  box.get(LeftSide).getData(),
                                  box.get(RightSide).getData(),
                                  0, rhsValues);
  delete[] values;
  delete[] rhsValues;
  delete[] entries;
} // end makeUnderlyingIdentity

void
Solver::makeInteriorEquations(const Counter level,
                              const Hierarchy& hier,
                              const HYPRE_SStructGrid& grid,
                              const HYPRE_SStructStencil& stencil)
  //_____________________________________________________________________
  // Function Solver::makeInteriorEquations~
  // Initialize the linear system equations (LHS matrix A, RHS vector b)
  // at the interior of all patches at this level.
  //_____________________________________________________________________
{
  funcPrint("Solver::makeLinearSystem()",FBegin);
  linePrint("=",50);
  cout_doing << "Adding interior equations to A, level = "
             << level << "\n";
  linePrint("=",50);
  const Counter& numDims   = _param->numDims;
  Counter stencilSize = hypre_SStructStencilSize(stencil);
  int* entries = new int[stencilSize];
  for (Counter entry = 0; entry < stencilSize; entry++) {
    entries[entry] = entry;
  }
  const Level* lev = hier._levels[level];
  const Vector<double>& h = lev->_meshSize;
  //const Vector<Counter>& resolution = lev->_resolution;
  Vector<double> offset = 0.5 * h;
  double cellVolume = h.prod();
  dbg.indent();
  for (Counter i = 0; i < lev->_patchList[MYID].size(); i++) {
    cout_doing << "At patch = " << i << "\n";
    // Add equations of interior cells of this patch to A
    Patch* patch = lev->_patchList[MYID][i];
    double* values    = new double[stencilSize * patch->_numCells];
    double* rhsValues = new double[patch->_numCells];
    double* solutionValues = new double[patch->_numCells];
    cout_doing << "Adding interior equations at Patch " << i
               << ", Extents = " << patch->_box << "\n";
    cout_doing << "Looping over cells in this patch:" << "\n";

    Counter cell = 0;
    for(Box::iterator iter = patch->_box.begin();
        iter != patch->_box.end(); ++iter, cell++ ) {
      cout_doing << "  sub = " << *iter << "\n";
      int offsetValues    = stencilSize * cell;
      int offsetRhsValues = cell;
      // Initialize the stencil values of this cell's equation to 0
      for (Counter entry = 0; entry < stencilSize; entry++) {
        values[offsetValues + entry] = 0.0;
      }

      // Compute RHS integral over the cell. Using the mid-point
      // rule, and assuming that xCell is also the centroid of the
      // cell.
      Vector<double> xCell = offset + (h * (*iter));;
      rhsValues[offsetRhsValues] = cellVolume * _param->rhs(xCell);

      // Assuming a constant initial guess
      solutionValues[offsetRhsValues] = 1234.56;
        
      // Loop over directions
      Counter entry = 1;
      for (Counter d = 0; d < numDims; d++) {
        double faceArea = cellVolume / h[d];
        for (BoxSide s = LeftSide; s <= RightSide; ++s) {
          cout_doing << "--- d = " << d
                     << " , s = " << s
                     << " , entry = " << entry
                     << " ---" << "\n";
          // Compute coordinates of:
          // This cell's center: xCell
          // The neighboring's cell data point: xNbhr
          // The face crossing between xCell and xNbhr: xFace
          Vector<double> xNbhr  = xCell;
            
          xNbhr[d] += s*h[d];

          cout_doing << "1) xCell = " << xCell
                     << ", xNbhr = " << xNbhr << "\n";

          if( (patch->getBoundaryType(d,s) == Patch::Domain) && 
              ((*iter)[d] == patch->_box.get(s)[d]) ) {
            // Cell near a domain boundary
              
            if (patch->getBC(d,s) == Patch::Dirichlet) {
              cout_doing << "Near Dirichlet boundary, update xNbhr"
                         << "\n";
              xNbhr[d] = xCell[d] + 0.5*s*h[d];
            } else {
              // TODO: put something in this loop?

              // Neumann B.C., xNbhr is outside the domain. We
              // assume that a, du/dn can be continously extended
              // outside the domain to make the B.C. a du/dn = rhsBC
              // meaningful using a central difference and a
              // harmonic avg of a over the line of that central
              // difference. Otherwise, go back to the Dirichlet
              // code at the expense of larger truncation errors
              // near these boundaries, that should not matter, as
              // in the Dirichlet case.
            }
                            
          } // end cell near domain boundary

          Vector<double> xFace = xCell;
          xFace[d] = 0.5*(xCell[d] + xNbhr[d]);
          cout_doing << "xCell = " << xCell
                     << ", xFace = " << xFace
                     << ", xNbhr = " << xNbhr << "\n";

          //--- Compute flux ---
          // Harmonic average of diffusion for this face 
          double a    = _param->harmonicAvg(xCell,xNbhr,xFace); 
          double diff = fabs(xNbhr[d] - xCell[d]);  // for FD approx of flux
          double flux = a * faceArea / diff;        // total flux thru face

          // Accumulate this flux's contribution to values
          // if we are not near a C/F boundary.
          // TODO: CHECK THIS!!!
          if (!((patch->getBoundaryType(d,s) == Patch::CoarseFine) &&
                ((*iter)[d] == patch->_box.get(s)[d]))) {
            values[offsetValues        ] += flux;
            values[offsetValues + entry] -= flux;
          }

          // If we are next to a domain boundary, eliminate boundary variable
          // from the linear system.
          if( (patch->getBoundaryType(d,s) == Patch::Domain) && 
              ((*iter)[d] == patch->_box.get(s)[d]) ) {
            // Cell near a domain boundary
            // Nbhr is at the boundary, eliminate it from values

            if (patch->getBC(d,s) == Patch::Dirichlet) {
              cout_doing << "Near Dirichlet boundary, eliminate nbhr, "
                         << "coef = " << values[offsetValues + entry]
                         << ", rhsBC = " << _param->rhsBC(xNbhr) << "\n";
              // Pass boundary value to RHS
              rhsValues[offsetRhsValues] -= 
                values[offsetValues + entry] * _param->rhsBC(xNbhr);

              values[offsetValues + entry] = 0.0; // Eliminate connection
              // TODO:
              // Add to rhsValues if this is a non-zero Dirichlet B.C. !!
            } else { // Neumann B.C.
              cout_doing << "Near Neumann boundary, eliminate nbhr"
                         << "\n";
              // TODO:
              // DO NOT ADD FLUX ABOVE, and add to rhsValues appropriately,
              // if this is a non-zero Neumann B.C.
            }
          }
          entry++;
        } // end for s
      } // end for d

	//======== BEGIN GOOD DEBUGGING CHECK =========
	// This will set the diagonal entry of this cell's equation
	// to cell so that we can compare our cell numbering with
	// Hypre's cell numbering within each patch.
	// Hypre does it like we do: first loop over x, then over y,
	// then over z.
	//        values[offsetValues] = cell;
	//======== END GOOD DEBUGGING CHECK =========

    } // end for cell
      
    printValues(patch->_box.volume(),stencilSize,
                values,rhsValues,solutionValues);

    // Add this patch's interior equations to the LHS matrix A 
    cout_doing << "Calling HYPRE_SStructMatrixSetBoxValues A" << "\n";
    HYPRE_SStructMatrixSetBoxValues(_A, level, 
                                    patch->_box.get(LeftSide).getData(),
                                    patch->_box.get(RightSide).getData(),
                                    0, stencilSize, entries, values);

    // Add this patch's interior RHS to the RHS vector b 
    cout_doing << "Calling HYPRE_SStructVectorSetBoxValues b" << "\n";
    HYPRE_SStructVectorSetBoxValues(_b, level,
                                    patch->_box.get(LeftSide).getData(),
                                    patch->_box.get(RightSide).getData(),
                                    0, rhsValues);

    // Add this patch's interior initial guess to the solution vector x 
    cout_doing << "Calling HYPRE_SStructVectorSetBoxValues x" << "\n";
    HYPRE_SStructVectorSetBoxValues(_x, level,
                                    patch->_box.get(LeftSide).getData(),
                                    patch->_box.get(RightSide).getData(),
                                    0, solutionValues);

    delete[] values;
    delete[] rhsValues;
    delete[] solutionValues;
  } // end for patch
  dbg.unindent();
  delete entries;
  funcPrint("Solver::makeInteriorEquations()",FEnd);
} // end makeInteriorEquations()

void
Solver::makeLinearSystem(const Hierarchy& hier,
                         const HYPRE_SStructGrid& grid,
                         const HYPRE_SStructStencil& stencil)
  //_____________________________________________________________________
  // Function Solver::makeLinearSystem~
  // Initialize the linear system: set up the values on the links of the
  // graph of the LHS matrix A and value of the RHS vector b at all
  // patches of all levels. Delete coarse data underlying fine patches.
  //_____________________________________________________________________
{
  const int numLevels = hier._levels.size();

  //======================================================================
  // Add structured equations (stencil-based) at the interior of
  // each patch at every level to the graph.
  // Eliminate boundary conditions at domain boundaries.
  //======================================================================
  linePrint("*",50);
  cout_doing << "Matrix structured (interior) equations" << "\n";
  linePrint("*",50);
  for (Counter level = 0; level < numLevels; level++) {
    serializeProcsBegin();
    makeInteriorEquations(level,hier,grid,stencil);
    serializeProcsEnd();
  } // end for level

    //======================================================================
    // Add to graph the unstructured part of the stencil connecting the
    // coarse and fine level at every C/F boundary (F->C connections at
    // this patch's outer boundaries, and C->F connections at all
    // applicable C/F boundaries of next-finer-level patches that lie
    // above this patch.
    //======================================================================
  serializeProcsBegin();
  linePrint("*",50);
  cout_doing << "Matrix unstructured (C/F) equations" << "\n";
  linePrint("*",50);
  for (Counter level = 0; level < numLevels; level++) {
    const Level* lev = hier._levels[level];
    dbg.indent();
    // Loop over patches of this proc
    for (Counter i = 0; i < lev->_patchList[MYID].size(); i++) {
      Patch* patch = lev->_patchList[MYID][i];
      linePrint("%",40);
      cout_doing << "Processing Patch" << "\n"
                 << *patch << "\n";
      linePrint("%",40);
      
      if (level > 0) {
        // If not at coarsest level,
        // loop over outer boundaries of this patch and add
        // fine-to-coarse connections
        linePrint("=",50);
        cout_doing << "Building fine-to-coarse connections" << "\n";
        linePrint("=",50);
        for (Counter d = 0; d < numDims; d++) {
          for (BoxSide s = LeftSide; s <= RightSide; ++s) {
            if (patch->getBoundaryType(d,s) == Patch::CoarseFine) {
              makeConnections(Matrix,hier,stencil,level,patch,d,s,FineToCoarse);
            } // end if boundary is CF interface
          } // end for s
        } // end for d
      }

      if (level < numLevels-1) {
        linePrint("=",50);
        cout_doing << "Building coarse-to-fine connections" 
                   << " Patch ID = " << patch->_patchID
                   << "\n";
        linePrint("=",50);
        //  const int numDims   = _param->numDims;
        const Vector<Counter>& refRat = hier._levels[level+1]->_refRat;
        dbg.indent();
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
          // Delete the underlying coarse cell equations (those under the
          // fine patch). Replace them by the identity operator.
          //===================================================================
          makeUnderlyingIdentity(level,stencil,coarseUnderFine);
          
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
                makeConnections(Matrix,hier,stencil,level,finePatch,d,s,CoarseToFine);

              } // end if CoarseFine boundary
            } // end for s
          } // end for d
        } // end for all fine patches that cover this patch
      } // end if (level < numLevels-1)
    } // end for i (patches)
    dbg.unindent();
  } // end for level
  serializeProcsEnd();

  funcPrint("Solver::makeLinearSystem()",FEnd);
} // end makeLinearSystem()

std::ostream&
operator << (std::ostream& os,
             const HypreDriverSStruct::CoarseFineViewpoint& v)
  // Write side s to the stream os.
{
  if      (v == Solver::CoarseToFine) os << "CoarseToFine";
  else if (v == Solver::FineToCoarse) os << "FineToCoarse";
  else os << "N/A";
  return os;
}

std::ostream&
operator << (std::ostream& os,
             const HypreDriverSStruct::ConstructionStatus& s)
{
  if      (s == Solver::Graph ) os << "Graph ";
  else if (s == Solver::Matrix) os << "Matrix";
  else os << "ST WRONG!!!";
  return os;
}

HypreDriver::Side& operator++(HypreDriver::BoxSide &s)
{
  return s = Side(s+2);
}

std::ostream&
operator << (std::ostream& os, const HypreDriver::Side& s)
  // Write side s to the stream os.
{
  if      (s == LeftSide ) os << "Left ";
  else if (s == RightSide) os << "Right";
  else os << "N/A";
  return os;
}

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

static void linePrint(const string& s, const unsigned int len)
{
  for (unsigned int i = 0; i < len; i++) {
    cout_doing << s;
  }
  cout_doing << "\n";
}

