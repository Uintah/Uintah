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

#include <Packages/Uintah/CCA/Components/Solvers/HypreDriver.h>
#include <Packages/Uintah/CCA/Components/Solvers/HypreSolverParams.h>
#include <Packages/Uintah/CCA/Components/Solvers/HypreGenericSolver.h>
#include <Packages/Uintah/CCA/Components/Solvers/MatrixUtil.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/NCVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCZVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Packages/Uintah/Core/Grid/Variables/Stencil7.h>
#include <Packages/Uintah/Core/Grid/Variables/VarTypes.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Packages/Uintah/Core/Exceptions/ConvergenceFailure.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/CCA/Ports/LoadBalancer.h>
#include <Core/Containers/Array1.h>
#include <Core/Containers/Array2.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Math/MiscMath.h>
#include <Core/Math/MinMax.h>
#include <Core/Thread/Time.h>
#include <Core/Util/DebugStream.h>
#include <iomanip>

using namespace Uintah;
//__________________________________
//  To turn on normal output
//  setenv SCI_DEBUG "HYPRE_DOING_COUT:+"

static DebugStream cout_doing("HYPRE_DOING_COUT", false);

/*_____________________________________________________________________
  class HypreDriverSStruct implementation common to all variable types
  _____________________________________________________________________*/

HypreDriverSStruct::~HypreDriverSStruct(void)
{
  // Destroy matrix, RHS, solution objects
  cout_doing << "Destroying SStruct matrix, RHS, solution objects" << "\n";
  HYPRE_SStructMatrixDestroy(_A_SStruct);
  HYPRE_SStructVectorDestroy(_b_SStruct);
  HYPRE_SStructVectorDestroy(_x_SStruct);
  
  // Destroy graph objects
  cout_doing << "Destroying Solver object" << "\n";
  HYPRE_SStructGraphDestroy(_graph_SStruct);
  
  // Destroying grid, stencil
  HYPRE_StructStencilDestroy(_stencil);
  HYPRE_StructGridDestroy(_grid);
}

template<class Types>
void
HypreDriverSStruct<Types>::solve(const ProcessorGroup* pg,
                          const PatchSubset* patches,
                          const MaterialSubset* matls,
                          DataWarehouse* old_dw,
                          DataWarehouse* new_dw,
                          Handle<HypreDriverSStruct<Types> >)
  /*_____________________________________________________________________
    Function HypreDriverSStruct::solve~
    Main solve function.
    _____________________________________________________________________*/
{
  typedef typename Types::sol_type sol_type;
  cout_doing << "HypreSolverAMR<Types>::solve()" << endl;

  DataWarehouse* A_dw = new_dw->getOtherDataWarehouse(which_A_dw);
  DataWarehouse* b_dw = new_dw->getOtherDataWarehouse(which_b_dw);
  DataWarehouse* guess_dw = new_dw->getOtherDataWarehouse(which_guess_dw);
    
  // Check parameter correctness
  cerr << "Checking arguments and parameters ... ";
  HypreGenericSolver::SolverType solverType =
    getSolverType(p->solverTitle);
  const int numLevels = new_dw->getGrid()->numLevels();
  if ((solverType == HypreGenericSolver::FAC) && (numLevels < 2)) {
    cerr << "\n\nFAC solver needs a 3D problem and at least 2 levels."
         << "\n";
    clean();
    exit(1);
  }

  for(int m = 0;m<matls->size();m++){
    int matl = matls->get(m);

    /* Construct Hypre linear system for the specific variable type
       and Hypre interface */
    HypreInterface hypreInterface(params);
    hypreInterface.makeLinearSystemStruct(interfaceType);
    
    /* Construct Hypre solver object that uses the hypreInterface we
       chose. Specific solver object is arbitrated in HypreGenericSolver
       according to param->solverType. */
    HypreGenericSolver::SolverType solverType =
      HypreGenericSolver::solverFromTitle(params->solverTitle);
    HypreGenericSolver* _hypreSolver =
      HypreGenericSolver::newSolver(solverType,_hypreInterface);

    /* Solve the linear system */
    double solve_start = Time::currentSeconds();
    _hypresolver->setup();  // Depends only on A
    _hypresolver->solve();  // Depends on A and b
    double solve_dt = Time::currentSeconds()-solve_start;

    /* Check if converged, print solve statistics */
    const HypreGenericSolver::Results& results = _hypreSolver->getResults();
    const double& finalResNorm = results->finalResNorm;
    if ((finalResNorm > params->tolerance) ||
        (finite(finalResNorm) == 0)) {
      if (params->restart){
        if(pg->myrank() == 0)
          cerr << "HypreSolver not converged in " << results.numIterations 
               << "iterations, final residual= " << finalResNorm
               << ", requesting smaller timestep\n";
        //new_dw->abortTimestep();
        //new_dw->restartTimestep();
      } else {
        throw ConvergenceFailure("HypreSolver variable: "
                                 +X_label->getName()+
                                 ",solver: "+params->solverTitle+
                                 ", preconditioner: "+params->precondTitle,
                                 num_iterations, final_res_norm,
                                 params->tolerance,__FILE__,__LINE__);
      }
    } // if (finalResNorm is ok)

      /* Get the solution x values back into Uintah */
    switch (_hypreInterface) {
    case HypreSolverParams::Struct:
      {
        getSolutionStruct(matl);
        break;
      }
    case HypreSolverParams::SStruct:
      {
        getSolutionSStruct(matl);
        break;
      }
    default:
      {
        throw InternalError("Unknown Hypre interface for getSolution: "
                            +hypreInterface,__FILE__, __LINE__);
      }
    } // end switch (_hypreInterface)

      /*-----------------------------------------------------------
       * Print the solution and other info
       *-----------------------------------------------------------*/
    linePrint("-",50);
    dbg0 << "Print the solution vector" << "\n";
    linePrint("-",50);
    solver->printSolution("output_x1");
    dbg0 << "Iterations = " << solver->_results.numIterations << "\n";
    dbg0 << "Final Relative Residual Norm = "
         << solver->_results.finalResNorm << "\n";
    dbg0 << "" << "\n";
      
    delete _hypreSolver;
    clear(); // Destroy Hypre objects

    double dt=Time::currentSeconds()-tstart;
    if(pg->myrank() == 0){
      cerr << "Solve of " << X_label->getName() 
           << " on level " << level->getIndex()
           << " completed in " << dt 
           << " seconds (solve only: " << solve_dt 
           << " seconds, " << num_iterations 
           << " iterations, residual=" << final_res_norm << ")\n";
    }
    tstart = Time::currentSeconds();
  } // for m (matls loop)
} // end solve() for


template<class Types>
void
HypreDriverSStruct<Types>::makeLinearSystem(const HypreInterface& interface)
  /*_____________________________________________________________________
    Function HypreDriverSStruct::makeLinearSystem~
    Arbitrate the construction of the linear system, depending on the
    Hypre interface.
    _____________________________________________________________________*/
{
  switch (interface) {
  case Struct:
    {
      makeLinearSystemStruct();
    }
  case SStruct:
    {
      makeLinearSystemStruct();
    }
  default:
    {
      throw InternalError("Unsupported Hypre interface for makeLinearSystem: "
                          +hypreInterface,__FILE__, __LINE__);
    }
  } // end switch (interface)
} // end makeLinearSystem()

template<class Types>
void HypreDriverSStruct::getSolution(const HypreInterface& interface)
  /*_____________________________________________________________________
    Function HypreDriverSStruct::getSolution~
    Arbitrate the fetching of the solution vector, depending on the
    Hypre interface.
    _____________________________________________________________________*/
{
  switch (interface) {
  case Struct:
    {
      getSolutionStruct();
    }
  case SStruct:
    {
      getSolutionStruct();
    }
  default:
    {
      throw InternalError("Unsupported Hypre interface for getSolution: "
                          +hypreInterface,__FILE__, __LINE__);
    }
  } // end switch (interface)
} // end getSolution()

  /*_____________________________________________________________________
    class HypreDriverSStruct implementation for CC variables, Struct interface
    _____________________________________________________________________*/

void
HypreDriverSStruct<CCTypes>::makeLinearSystemStruct(void)
  /*_____________________________________________________________________
    Function HypreDriverSStruct<CCTypes>::makeLinearSystemStruct
    Construct the linear system for CC variables (e.g. pressure),
    for the Hypre Struct interface.
    _____________________________________________________________________*/
{
  ASSERTEQ(sizeof(Stencil7), 7*sizeof(double));
  double tstart = Time::currentSeconds();

  // Setup matrix
  HYPRE_StructGrid grid;
  HYPRE_StructGridCreate(pg->getComm(), 3, &grid);

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    Patch::VariableBasis basis =
      Patch::translateTypeToBasis(sol_type::getTypeDescription()
                                  ->getType(), true);
    IntVector ec = params->getSolveOnExtraCells() ?
      IntVector(0,0,0) : -level->getExtraCells();
    IntVector l = patch->getLowIndex(basis, ec);
    IntVector h1 = patch->getHighIndex(basis, ec)-IntVector(1,1,1);

    HYPRE_StructGridSetExtents(grid, l.get_pointer(), h1.get_pointer());
  }
  HYPRE_StructGridAssemble(grid);

  // Create the stencil
  HYPRE_StructStencil stencil;
  if(params->symmetric){
    HYPRE_StructStencilCreate(3, 4, &stencil);
    int offsets[4][3] = {{0,0,0},
                         {-1,0,0},
                         {0,-1,0},
                         {0,0,-1}};
    for(int i=0;i<4;i++)
      HYPRE_StructStencilSetElement(stencil, i, offsets[i]);
  } else {
    HYPRE_StructStencilCreate(3, 7, &stencil);
    int offsets[7][3] = {{0,0,0},
                         {1,0,0}, {-1,0,0},
                         {0,1,0}, {0,-1,0},
                         {0,0,1}, {0,0,-1}};
    for(int i=0;i<7;i++)
      HYPRE_StructStencilSetElement(stencil, i, offsets[i]);
  }

  // Create the matrix
  HYPRE_StructMatrix HA;
  HYPRE_StructMatrixCreate(pg->getComm(), grid, stencil, &HA);
  HYPRE_StructMatrixSetSymmetric(HA, params->symmetric);
  int ghost[] = {1,1,1,1,1,1};
  HYPRE_StructMatrixSetNumGhost(HA, ghost);
  HYPRE_StructMatrixInitialize(HA);

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    // Get the data
    CCTypes::matrix_type A;
    A_dw->get(A, A_label, matl, patch, Ghost::None, 0);

    Patch::VariableBasis basis =
      Patch::translateTypeToBasis(sol_type::getTypeDescription()
                                  ->getType(), true);
    IntVector ec = params->getSolveOnExtraCells() ?
      IntVector(0,0,0) : -level->getExtraCells();
    IntVector l = patch->getLowIndex(basis, ec);
    IntVector h = patch->getHighIndex(basis, ec);

    // Feed it to Hypre
    if(params->symmetric){
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
          HYPRE_StructMatrixSetBoxValues(HA,
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
          HYPRE_StructMatrixSetBoxValues(HA,
                                         ll.get_pointer(),
                                         hh.get_pointer(),
                                         7, stencil_indices,
                                         const_cast<double*>(values));
        }
      }
    }
  }
  HYPRE_StructMatrixAssemble(HA);

  // Create the rhs
  HYPRE_StructVector HB;
  HYPRE_StructVectorCreate(pg->getComm(), grid, &HB);
  HYPRE_StructVectorInitialize(HB);

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    // Get the data
    CCTypes::const_type B;
    b_dw->get(B, B_label, matl, patch, Ghost::None, 0);

    Patch::VariableBasis basis =
      Patch::translateTypeToBasis(sol_type::getTypeDescription()
                                  ->getType(), true);
    IntVector ec = params->getSolveOnExtraCells() ?
      IntVector(0,0,0) : -level->getExtraCells();
    IntVector l = patch->getLowIndex(basis, ec);
    IntVector h = patch->getHighIndex(basis, ec);

    // Feed it to Hypre
    for(int z=l.z();z<h.z();z++){
      for(int y=l.y();y<h.y();y++){
        const double* values = &B[IntVector(l.x(), y, z)];
        IntVector ll(l.x(), y, z);
        IntVector hh(h.x()-1, y, z);
        HYPRE_StructVectorSetBoxValues(HB,
                                       ll.get_pointer(),
                                       hh.get_pointer(),
                                       const_cast<double*>(values));
      }
    }
  }
  HYPRE_StructVectorAssemble(HB);

  // Create the solution vector
  HYPRE_StructVector HX;
  HYPRE_StructVectorCreate(pg->getComm(), grid, &HX);
  HYPRE_StructVectorInitialize(HX);

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    if(guess_label){
      CCTypes::const_type X;
      guess_dw->get(X, guess_label, matl, patch, Ghost::None, 0);

      // Get the initial guess
      Patch::VariableBasis basis =
        Patch::translateTypeToBasis(sol_type::getTypeDescription()
                                    ->getType(), true);
      IntVector ec = params->getSolveOnExtraCells() ?
        IntVector(0,0,0) : -level->getExtraCells();
      IntVector l = patch->getLowIndex(basis, ec);
      IntVector h = patch->getHighIndex(basis, ec);

      // Feed it to Hypre
      for(int z=l.z();z<h.z();z++){
        for(int y=l.y();y<h.y();y++){
          const double* values = &X[IntVector(l.x(), y, z)];
          IntVector ll(l.x(), y, z);
          IntVector hh(h.x()-1, y, z);
          HYPRE_StructVectorSetBoxValues(HX,
                                         ll.get_pointer(),
                                         hh.get_pointer(),
                                         const_cast<double*>(values));
        }
      }
    }  // initialGuess
  } // patch loop
  HYPRE_StructVectorAssemble(HX);
} // end HypreDriverSStruct<CCTypes>::makeLinearSystemStruct()


void HypreDriverSStruct<CCTypes>::getSolutionStruct(void)
  /*_____________________________________________________________________
    Function HypreDriverSStruct<CCTypes>::makeLinearSystemStruct
    Get the solution vector for CC variables (e.g. pressure),
    for the Hypre Struct interface.
    _____________________________________________________________________*/
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    Patch::VariableBasis basis =
      Patch::translateTypeToBasis(sol_type::getTypeDescription()
                                  ->getType(), true);
    IntVector ec = params->getSolveOnExtraCells() ?
      IntVector(0,0,0) : -level->getExtraCells();
    IntVector l = patch->getLowIndex(basis, ec);
    IntVector h = patch->getHighIndex(basis, ec);
    CellIterator iter(l, h);

    sol_type Xnew;
    if(modifies_x)
      new_dw->getModifiable(Xnew, X_label, matl, patch);
    else
      new_dw->allocateAndPut(Xnew, X_label, matl, patch);
	
    // Get the solution back from hypre
    for(int z=l.z();z<h.z();z++){
      for(int y=l.y();y<h.y();y++){
        const double* values = &Xnew[IntVector(l.x(), y, z)];
        IntVector ll(l.x(), y, z);
        IntVector hh(h.x()-1, y, z);
        HYPRE_StructVectorGetBoxValues(HX,
                                       ll.get_pointer(),
                                       hh.get_pointer(),
                                       const_cast<double*>(values));
      }
    }
  }
} // end HypreDriverSStruct<CCTypes>::getSolutionStruct()

  /*_____________________________________________________________________
    class HypreDriverSStruct implementation for CC variables, SStruct interface
    _____________________________________________________________________*/

void
Solver::initializeSStruct(void)
{
  funcPrint("Solver::initialize()",FBegin);
  _requiresPar =
    (((_solverID >= 20) && (_solverID <= 30)) ||
     ((_solverID >= 40) && (_solverID < 60)));
  dbg0 << "requiresPar = " << _requiresPar << "\n";
    
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
  dbg0 << "Print out the system and initial guess" << "\n";
  linePrint("-",50);
  solver->printMatrix("output_A");
  solver->printRHS("output_b");
  solver->printSolution("output_x0");



  funcPrint("Solver::initialize()",FEnd);
}
  
void clearStruct(void)
{
  HYPRE_StructMatrixDestroy(HA);
  HYPRE_StructVectorDestroy(HB);
  HYPRE_StructVectorDestroy(HX);
  HYPRE_StructStencilDestroy(stencil);
  HYPRE_StructGridDestroy(grid);
}

void clearSStruct(void)
{
  /*-----------------------------------------------------------
   * Finalize things
   *-----------------------------------------------------------*/
  serializeProcsBegin();
  linePrint("-",50);
  cerr << "Finalize things" << "\n";
  linePrint("-",50);
    
  /* Destroy grid objects */
  cerr << "Destroying grid objects" << "\n";
  HYPRE_SStructGridDestroy(grid);
    
  /* Destroy stencil objects */
  cerr << "Destroying stencil objects" << "\n";
  HYPRE_SStructStencilDestroy(stencil);
}

void
makeGrid(const Param* param,
         const Hierarchy& hier,
         HYPRE_SStructGrid& grid)
  /*_____________________________________________________________________
    Function makeGrid:
    Create an empty Hypre grid object "grid" from our hierarchy hier,
    and add all patches from this proc to it.
    _____________________________________________________________________*/
{
  // TODO:
  // Read Uintah level & patch hierarchy
  // Get boundary information of each patch (Domain/CoarseFine/NA)
  // Get a list of all fine patches over a coarse patch (see impICE)
  serializeProcsBegin();
  funcPrint("makeGrid()",FBegin);
  HYPRE_SStructVariable vars[NUM_VARS] =
    {HYPRE_SSTRUCT_VARIABLE_CELL}; // We use cell centered vars
  const Counter numDims   = param->numDims;
  const Counter numLevels = hier._levels.size();
  serializeProcsEnd();

  /* Create an empty grid in numDims dimensions with # parts = numLevels. */
  HYPRE_SStructGridCreate(MPI_COMM_WORLD, numDims, numLevels, &grid);

  serializeProcsBegin();
  /* Add the patches that this proc owns at all levels to grid */
  dbg0 << "Adding hier patches to HYPRE grid:" << "\n";
  for (Counter level = 0; level < numLevels; level++) {
    Level* lev = hier._levels[level];
    dbg.setLevel(1);
    dbg << "Level " << level 
        << ", meshSize = " << lev->_meshSize[0]
        << ", resolution = " << lev->_resolution << "\n";
    dbg.indent();
    for (Counter i = 0; i < lev->_patchList[MYID].size(); i++) {
      Patch* patch = lev->_patchList[MYID][i];
      /* Add this patch to the grid */
      HYPRE_SStructGridSetExtents(grid, level,
                                  patch->_box.get(Left ).getData(),
                                  patch->_box.get(Right).getData());
      HYPRE_SStructGridSetVariables(grid, level, NUM_VARS, vars);
      dbg.setLevel(1);
      dbg << "  Patch " << setw(2) << left << i
          << ", ID "    << setw(2) << left << patch->_patchID << " ";
      dbg << patch->_box;
      dbg << "\n";
    }
    dbg.unindent();
  }

  /*
    Assemble the grid; this is a collective call that synchronizes
    data from all processors. On exit from this function, the grid is
    ready.
  */
  dbg.setLevel(10);
  dbg << "Before HYPRE Grid Assemble call" << "\n";
  serializeProcsEnd();

  HYPRE_SStructGridAssemble(grid);

  serializeProcsBegin();
  dbg.setLevel(1);
  dbg0 << "" << "\n";
  dbg0 << "Assembled grid, "
       << "num parts " << hypre_SStructGridNParts(grid) << "\n";
  dbg0 << "\n";
  funcPrint("makeGrid()",FEnd);
  serializeProcsEnd();
} // end makeGrid()


void
makeStencil(const Param* param,
            const Hierarchy& hier,
            HYPRE_SStructStencil& stencil)
  /*_____________________________________________________________________
    Function makeStencil:
    Initialize the Hypre stencil with a 5-point stencil (in d dimensions).
    Create Hypre stencil object "stencil" on output.
    _____________________________________________________________________*/
{
  funcPrint("makeStencil()",FBegin);
  serializeProcsBegin();
  const Counter numDims   = param->numDims;
  Counter               stencilSize = 2*numDims+1;
  Vector< Vector<int> > stencil_offsets;
  serializeProcsEnd();
  
  /* Create an empty stencil */
  HYPRE_SStructStencilCreate(numDims, stencilSize, &stencil);
  
  serializeProcsBegin();
  /*
    The following code is for general dimension.
    We use a 5-point FD discretization to L = -Laplacian in this example.
    stencil_offsets specifies the non-zero pattern of the stencil;
    Its entries are also defined here and assumed to be constant over the
    structured mesh. If not, define it later during matrix setup.
  */
  dbg.setLevel(10);
  dbg << "stencilSize = " << stencilSize << ",  numDims = " << numDims << "\n";
  stencil_offsets.resize(0,stencilSize);
  /* Order them as follows: center, xminus, xplus, yminus, yplus, etc. */
  /* Central coeffcient */
  Counter entry = 0;
  stencil_offsets[entry].resize(0,numDims);
  for (Counter dim = 0; dim < numDims; dim++) {
    dbg << "Init entry = " << setw(2) << left << entry
        << ", dim = " << dim << "\n";
    stencil_offsets[entry][dim] = 0;
  }
  for (Counter dim = 0; dim < numDims; dim++) {
    for (int s = Left; s <= Right; s += 2) {
      entry++;
      dbg << "Init    entry = " << setw(2) << left << entry
          << ", dim = " << dim << "\n";
      stencil_offsets[entry].resize(0,numDims);
      dbg.indent();
      for (Counter d = 0; d < numDims; d++) {
        dbg << "d = " << d
            << "  size = " << stencil_offsets[entry].getLen() << "\n";
        stencil_offsets[entry][d] = 0;
      }
      dbg.unindent();
      dbg << "Setting entry = " << setw(2) << left << entry
          << ", dim = " << dim << "\n";
      stencil_offsets[entry][dim] = s;
    }
  }
  
  /* Add stencil entries */
  dbg0 << "Stencil offsets:" << "\n";
  dbg0.indent();
  for (entry = 0; entry < stencilSize; entry++) {
    HYPRE_SStructStencilSetEntry(stencil, entry,
                                 stencil_offsets[entry].getData(), 0);
    dbg0 << "entry = " << entry
         << "  stencil_offsets = " << stencil_offsets[entry] << "\n";
  }
  dbg0.unindent();
  funcPrint("makeStencil()",FEnd);
  serializeProcsEnd();
} // end makeStencil()

void
Solver::initializeData(const Hierarchy& hier,
                       const HYPRE_SStructGrid& grid)
{
  funcPrint("Solver::initializeData()",FBegin);

  // Create an empty matrix with the graph non-zero pattern
  HYPRE_SStructMatrixCreate(MPI_COMM_WORLD, _graph, &_A);
  dbg0 << "Created empty SStructMatrix" << "\n";
  // Initialize RHS vector b and solution vector x
  dbg << "Create empty b,x" << "\n";
  HYPRE_SStructVectorCreate(MPI_COMM_WORLD, grid, &_b);
  dbg << "Done b" << "\n";
  HYPRE_SStructVectorCreate(MPI_COMM_WORLD, grid, &_x);
  dbg << "Done x" << "\n";

  // If using AMG, set (A,b,x)'s object type to ParCSR now
  if (_requiresPar) {
    dbg0 << "Matrix object type set to HYPRE_PARCSR" << "\n";
    HYPRE_SStructMatrixSetObjectType(_A, HYPRE_PARCSR);
    dbg << "Vector object type set to HYPRE_PARCSR" << "\n";
    HYPRE_SStructVectorSetObjectType(_b, HYPRE_PARCSR);
    dbg << "Done b" << "\n";
    HYPRE_SStructVectorSetObjectType(_x, HYPRE_PARCSR);
    dbg << "Done x" << "\n";
  }
  dbg << "Init A" << "\n";
  HYPRE_SStructMatrixInitialize(_A);
  dbg << "Init b,x" << "\n";
  HYPRE_SStructVectorInitialize(_b);
  dbg << "Done b" << "\n";
  HYPRE_SStructVectorInitialize(_x);
  dbg << "Done x" << "\n";

  funcPrint("Solver::initializeData()",FEnd);
}

void
Solver::assemble(void)
{
  dbg << "Solver::assemble() end" << "\n";
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
  dbg << "Solver::assemble() begin" << "\n";
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
  dbg0 << "Building connections" 
       << "  level = " << level 
       << "  patch =\n" << *patch << "\n"
       << "  status = " << status
       << "  viewpoint = " << viewpoint << "\n"
       << "  Face d = " << d
       << " , s = " << s << "\n";
  linePrint("=",50);
  dbg.setLevel(2);
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

  dbg.setLevel(2);
  dbg << "coarseLevel = " << coarseLevel << "\n";
  dbg << "fineLevel   = " << fineLevel   << "\n";
  // Loop over the C/F coarse cells and add connections
  for(Box::iterator coarse_iter = faceCoarseBox.begin();
      coarse_iter != faceCoarseBox.end(); ++coarse_iter) {
    // Compute the part fineCellFace of the fine face that directly
    // borders this coarse cell.
    Vector<int> cellFaceLower;
    if (s == Left) {
      Vector<int> coarseCellOverFineCells = *coarse_iter;
      coarseCellOverFineCells[d] -= s;
      cellFaceLower = coarseCellOverFineCells * refRat;
    } else { // s == Right
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
      dbg.setLevel(2);
      dbg << "Coarse cell: " << *coarse_iter << "\n";
      dbg << "Fine   cell: " << *fine_iter   << "\n";
      if (status == Graph) {
        if (viewpoint == FineToCoarse) {
          //====================================================
          // Add F->C connections to graph
          //====================================================
          dbg << "Adding F->C connection to graph" << "\n";
          HYPRE_SStructGraphAddEntries(_graph,
                                       fineLevel,(*fine_iter).getData(),0,
                                       coarseLevel,(*coarse_iter).getData(),0);
          dbg << "HYPRE call done" << "\n";
        } else { // viewpoint == CoarseToFine
          //====================================================
          // Add C->F connections to graph
          //====================================================
          dbg << "Adding C->F connection to graph" << "\n";
          HYPRE_SStructGraphAddEntries(_graph,
                                       coarseLevel,(*coarse_iter).getData(),0,
                                       fineLevel,(*fine_iter).getData(),0);
          dbg << "HYPRE call done" << "\n";
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
        dbg.setLevel(3);
        dbg << "alpha = " << alpha << "\n";
        Vector<double> xFace(0,numDims);
        for (Counter dd = 0; dd < numDims; dd++) {
          xFace[dd] = alpha*xCell[dd] + (1-alpha)*xNbhr[dd];
        }
        dbg << "xCell = " << xCell
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
        dbg << "C/F flux = " << flux
            << "   a = " << a
            << " face = " << faceArea
            << " diff = " << diff << "\n";
        
        if (viewpoint == FineToCoarse) {
          //====================================================
          // Compute matrix entries of the F->C graph connections
          //====================================================
          dbg << "Adding F->C flux to matrix" << "\n";
                    
          //########################################################
          // Add the flux to the fine cell equation - stencil part
          //########################################################
          const int numStencilEntries = 1;
          int stencilEntries[numStencilEntries] = {0};
          double stencilValues[numStencilEntries] = {flux};
          dbg << "Calling HYPRE_SStructMatrixAddToValues A (stencil)" << "\n";
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
            for (Side ss = Left; ss <= Right; ++ss) {
              if ((patch->getBoundaryType(dd,ss) == Patch::CoarseFine) &&
                  ((*fine_iter)[dd] == patch->_box.get(ss)[dd])) {
                entry++;
              }
            }
          }
          if ((s == Right) &&
              (patch->getBoundaryType(d,s) == Patch::CoarseFine) &&
              ((*fine_iter)[d] == patch->_box.get(Left)[d])) {
            entry++;
          }
          dbg.setLevel(2);
          dbg << "entry (Fine cell -> coarse) = " << entry << "\n";
          int graphEntries[numGraphEntries] = {entry};
          double graphValues[numGraphEntries] = {-flux};
          dbg << "Calling HYPRE_SStructMatrixAddToValues A (graph)" << "\n";
          HYPRE_SStructMatrixAddToValues(_A,
                                         fineLevel,(*fine_iter).getData(),
                                         0,numGraphEntries,
                                         graphEntries,
                                         graphValues);
        } else { // viewpoint == CoarseToFine
          //====================================================
          // Compute matrix entries of the C->F graph connections
          //====================================================
          dbg << "Adding C->F flux to matrix" << "\n";

          //########################################################
          // Add the C/F flux coarse cell equation - stencil part
          //########################################################
          const int numCoarseStencilEntries = 1;
          int coarseStencilEntries[numCoarseStencilEntries] = {0};
          double coarseStencilValues[numCoarseStencilEntries] = {flux};
          dbg << "Calling HYPRE_SStructMatrixAddToValues A (stencil)" << "\n";
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
          dbg << "fineCell = " << fineCell << "\n";
          dbg << "entry (coarse cell -> fine cell) = "
              << stencilSize+fineCell << "\n";
          double coarseGraphValues[numCoarseGraphEntries] = {-flux};
          dbg << "Calling HYPRE_SStructMatrixAddToValues A (graph)" << "\n";
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
            dbg << "Removing C/C flux connections from matrix" << "\n";
            Side s2 = Side(-s);
            dbg << "      s = " << s
                << ", s2 = " << s2 << "\n";
            xCell = coarseOffset + (coarseH * (*coarse_iter));
            xNbhr    = xCell;
            xNbhr[d] += s2*coarseH[d];
            xFace    = xCell;
            xFace[d] = 0.5*(xCell[d] + xNbhr[d]);
            dbg << "xCell = " << xCell
                << ", xFace = " << xFace
                << ", xNbhr = " << xNbhr << "\n";
            
            // Compute the harmonic average of the diffusion
            // coefficient
            double a    = 1.0; // Assumed constant a for now
            double diff = fabs(xNbhr[d] - xCell[d]);
            double flux = a * coarseFaceArea / diff;
            dbg << "      C/F flux = " << flux
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
Solver::makeGraph(const Hierarchy& hier,
                  const HYPRE_SStructGrid& grid,
                  const HYPRE_SStructStencil& stencil)
  //_____________________________________________________________________
  // Function Solver::makeGraph~
  // Initialize the graph from stencils (interior equations) and C/F
  // interface connections. Create Hypre graph object "_graph" on output.
  //_____________________________________________________________________
{
  serializeProcsBegin();
  funcPrint("Solver::makeGraph()",FBegin);
  const Counter numDims = _param->numDims;
  //-----------------------------------------------------------
  // Set up the graph
  //-----------------------------------------------------------
  linePrint("#",60);
  dbg0 << "Set up the graph" << "\n";
  linePrint("#",60);
  serializeProcsEnd();

  // Create an empty graph
  HYPRE_SStructGraphCreate(MPI_COMM_WORLD, grid, &_graph);
  // If using AMG, set graph's object type to ParCSR now
  if (_requiresPar) {
    dbg0 << "graph object type set to HYPRE_PARCSR" << "\n";
    HYPRE_SStructGraphSetObjectType(_graph, HYPRE_PARCSR);
  }

  //  const int numDims   = _param->numDims;
  const int numLevels = hier._levels.size();
  //======================================================================
  //  Add structured equations (stencil-based) at the interior of
  //  each patch at every level to the graph.
  //======================================================================
  serializeProcsBegin();
  linePrint("*",50);
  dbg0 << "Graph structured (interior) connections" << "\n";
  linePrint("*",50);
  for (Counter level = 0; level < numLevels; level++) {
    dbg0 << "  Initializing graph stencil at level " << level
         << " of " << numLevels << "\n";
    HYPRE_SStructGraphSetStencil(_graph, level, 0, stencil);
  }
  serializeProcsEnd();

  //======================================================================
  // Add to graph the unstructured part of the stencil connecting the
  // coarse and fine level at every C/F boundary (F->C connections at
  // this patch's outer boundaries, and C->F connections at all
  // applicable C/F boundaries of next-finer-level patches that lie
  // above this patch.
  //======================================================================
  serializeProcsBegin();
  linePrint("*",50);
  dbg0 << "Graph unstructured (C/F) connections" << "\n";
  linePrint("*",50);
  for (Counter level = 0; level < numLevels; level++) {
    dbg.setLevel(1);
    const Level* lev = hier._levels[level];
    dbg.indent();
    // Loop over patches of this proc
    for (Counter i = 0; i < lev->_patchList[MYID].size(); i++) {
      Patch* patch = lev->_patchList[MYID][i];
      dbg.setLevel(2);
      linePrint("$",40);
      dbg << "Processing Patch" << "\n"
          << *patch << "\n";
      linePrint("$",40);
      
      if (level > 0) {
        // If not at coarsest level,
        // loop over outer boundaries of this patch and add
        // fine-to-coarse connections
        linePrint("=",50);
        dbg0 << "Building fine-to-coarse connections" << "\n";
        linePrint("=",50);
        for (Counter d = 0; d < numDims; d++) {
          for (Side s = Left; s <= Right; ++s) {
            if (patch->getBoundaryType(d,s) == Patch::CoarseFine) {
              dbg.setLevel(2);
              dbg << "boundary is " << patch->getBoundaryType(d,s) << "\n";
              makeConnections(Graph,hier,stencil,level,patch,d,s,FineToCoarse);
            } // end if boundary is CF interface
          } // end for s
        } // end for d
      }

      if (level < numLevels-1) {
        linePrint("=",50);
        dbg0 << "Building coarse-to-fine connections" << "\n";
        linePrint("=",50);
        //  const int numDims   = _param->numDims;
        const Vector<Counter>& refRat = hier._levels[level+1]->_refRat;
        dbg.indent();
        // List of fine patches covering this patch
        vector<Patch*> finePatchList = hier.finePatchesOverMe(*patch);
        Box coarseRefined(patch->_box.get(Left) * refRat,
                          (patch->_box.get(Right) + 1) * refRat - 1);
        dbg << "coarseRefined " << coarseRefined << "\n";
    
        //===================================================================
        // Loop over next-finer level patches that cover this patch
        //===================================================================
        for (vector<Patch*>::iterator iter = finePatchList.begin();
             iter != finePatchList.end(); ++iter) {
          //===================================================================
          // Compute relevant boxes at coarse and fine levels
          //===================================================================
          Patch* finePatch = *iter;
          dbg.setLevel(3);
          dbg << "Considering patch "
              << "ID=" << setw(2) << left << finePatch->_patchID << " "
              << "owner=" << setw(2) << left << finePatch->_procID << " "
              << finePatch->_box << " ..." << "\n";
          // Intersection of fine and coarse patches in fine-level subscripts
          Box fineIntersect = coarseRefined.intersect(finePatch->_box);
          // Intersection of fine and coarse patches in coarse-level subscripts
          Box coarseUnderFine(fineIntersect.get(Left) / refRat, 
                              fineIntersect.get(Right) / refRat);
          dbg.setLevel(2);
          dbg << "fineIntersect   = " << fineIntersect   << "\n";
          dbg << "coarseUnderFine = " << coarseUnderFine << "\n";
          //===================================================================
          // Delete the underlying coarse cell equations (those under the
          // fine patch). Replace them by the identity operator.
          //===================================================================
          // No need to change the graph; only the matrix changes.
          //makeUnderlyingIdentity(level,stencil,coarseUnderFine);
          
          //===================================================================
          // Loop over coarse-to-fine internal boundaries of the fine patch;
          // add C-to-F connections; delete the old C-to-coarseUnderFine
          // connections.
          //===================================================================
          for (Counter d = 0; d < numDims; d++) {
            for (Side s = Left; s <= Right; ++s) {
              if (finePatch->getBoundaryType(d,s) == Patch::CoarseFine) {
                dbg << "fine boundary is " << finePatch->getBoundaryType(d,s) << "\n";
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
                makeConnections(Graph,hier,stencil,level,finePatch,d,s,CoarseToFine);

              } // end if CoarseFine boundary
            } // end for s
          } // end for d
        } // end for all fine patches that cover this patch
      } // end if (level < numLevels-1)
    } // end for i (patches)
    dbg.unindent();
  } // end for level
  serializeProcsEnd();
  
  // Assemble the graph
  HYPRE_SStructGraphAssemble(_graph);
  dbg << "Assembled graph, nUVentries = "
      << hypre_SStructGraphNUVEntries(_graph) << "\n";
  funcPrint("Solver::makeGraph()",FEnd);
} // end makeGraph()

void 
Solver::makeUnderlyingIdentity(const Counter level,
                               const HYPRE_SStructStencil& stencil,
                               const Box& coarseUnderFine)
  // Replace the matrix equations for the underlying coarse box
  // with the identity matrix.
{
  dbg.setLevel(2);
  dbg << "Putting identity on underlying coarse data" << "\n"
      << "coarseUnderFine " << coarseUnderFine << "\n";
  Counter stencilSize = hypre_SStructStencilSize(stencil);
  int* entries = new int[stencilSize];
  for (Counter entry = 0; entry < stencilSize; entry++) {
    entries[entry] = entry;
  }
  const Counter numCoarseCells = coarseUnderFine.volume();
  double* values    = new double[stencilSize * numCoarseCells];
  double* rhsValues = new double[numCoarseCells];

  dbg.setLevel(3);
  dbg << "Looping over cells in coarse underlying box:" 
      << "\n";
  Counter cell = 0;
  for (Box::iterator coarse_iter = coarseUnderFine.begin();
       coarse_iter != coarseUnderFine.end(); ++coarse_iter, cell++) {
    dbg.setLevel(3);
    dbg << "cell = " << cell << " " << *coarse_iter << "\n";
    
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
  dbg0 << "Calling HYPRE_SStructMatrixSetBoxValues A" << "\n";
  HYPRE_SStructMatrixSetBoxValues(_A, level,
                                  box.get(Left).getData(),
                                  box.get(Right).getData(),
                                  0, stencilSize, entries, values);
  dbg0 << "Calling HYPRE_SStructVectorSetBoxValues b" << "\n";
  HYPRE_SStructVectorSetBoxValues(_b, level, 
                                  box.get(Left).getData(),
                                  box.get(Right).getData(),
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
  dbg0 << "Adding interior equations to A, level = "
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
    dbg0 << "At patch = " << i << "\n";
    // Add equations of interior cells of this patch to A
    Patch* patch = lev->_patchList[MYID][i];
    double* values    = new double[stencilSize * patch->_numCells];
    double* rhsValues = new double[patch->_numCells];
    double* solutionValues = new double[patch->_numCells];
    dbg.setLevel(3);
    dbg << "Adding interior equations at Patch " << i
        << ", Extents = " << patch->_box << "\n";
    dbg << "Looping over cells in this patch:" << "\n";

    Counter cell = 0;
    for(Box::iterator iter = patch->_box.begin();
        iter != patch->_box.end(); ++iter, cell++ ) {
      dbg << "  sub = " << *iter << "\n";
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
        for (Side s = Left; s <= Right; ++s) {
          dbg << "--- d = " << d
              << " , s = " << s
              << " , entry = " << entry
              << " ---" << "\n";
          // Compute coordinates of:
          // This cell's center: xCell
          // The neighboring's cell data point: xNbhr
          // The face crossing between xCell and xNbhr: xFace
          Vector<double> xNbhr  = xCell;
            
          xNbhr[d] += s*h[d];

          dbg << "1) xCell = " << xCell
              << ", xNbhr = " << xNbhr << "\n";

          if( (patch->getBoundaryType(d,s) == Patch::Domain) && 
              ((*iter)[d] == patch->_box.get(s)[d]) ) {
            // Cell near a domain boundary
              
            if (patch->getBC(d,s) == Patch::Dirichlet) {
              dbg << "Near Dirichlet boundary, update xNbhr"
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
          dbg << "xCell = " << xCell
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
              dbg << "Near Dirichlet boundary, eliminate nbhr, "
                  << "coef = " << values[offsetValues + entry]
                  << ", rhsBC = " << _param->rhsBC(xNbhr) << "\n";
              // Pass boundary value to RHS
              rhsValues[offsetRhsValues] -= 
                values[offsetValues + entry] * _param->rhsBC(xNbhr);

              values[offsetValues + entry] = 0.0; // Eliminate connection
              // TODO:
              // Add to rhsValues if this is a non-zero Dirichlet B.C. !!
            } else { // Neumann B.C.
              dbg << "Near Neumann boundary, eliminate nbhr"
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
    dbg0 << "Calling HYPRE_SStructMatrixSetBoxValues A" << "\n";
    HYPRE_SStructMatrixSetBoxValues(_A, level, 
                                    patch->_box.get(Left).getData(),
                                    patch->_box.get(Right).getData(),
                                    0, stencilSize, entries, values);

    // Add this patch's interior RHS to the RHS vector b 
    dbg0 << "Calling HYPRE_SStructVectorSetBoxValues b" << "\n";
    HYPRE_SStructVectorSetBoxValues(_b, level,
                                    patch->_box.get(Left).getData(),
                                    patch->_box.get(Right).getData(),
                                    0, rhsValues);

    // Add this patch's interior initial guess to the solution vector x 
    dbg0 << "Calling HYPRE_SStructVectorSetBoxValues x" << "\n";
    HYPRE_SStructVectorSetBoxValues(_x, level,
                                    patch->_box.get(Left).getData(),
                                    patch->_box.get(Right).getData(),
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
  serializeProcsBegin();
  funcPrint("Solver::makeLinearSystem()",FBegin);
  const int numDims   = _param->numDims;
  const int numLevels = hier._levels.size();
  serializeProcsEnd();

  //======================================================================
  // Add structured equations (stencil-based) at the interior of
  // each patch at every level to the graph.
  // Eliminate boundary conditions at domain boundaries.
  //======================================================================
  linePrint("*",50);
  dbg0 << "Matrix structured (interior) equations" << "\n";
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
  dbg0 << "Matrix unstructured (C/F) equations" << "\n";
  linePrint("*",50);
  for (Counter level = 0; level < numLevels; level++) {
    dbg.setLevel(1);
    const Level* lev = hier._levels[level];
    dbg.indent();
    // Loop over patches of this proc
    for (Counter i = 0; i < lev->_patchList[MYID].size(); i++) {
      Patch* patch = lev->_patchList[MYID][i];
      dbg.setLevel(2);
      linePrint("%",40);
      dbg << "Processing Patch" << "\n"
          << *patch << "\n";
      linePrint("%",40);
      
      if (level > 0) {
        // If not at coarsest level,
        // loop over outer boundaries of this patch and add
        // fine-to-coarse connections
        linePrint("=",50);
        dbg0 << "Building fine-to-coarse connections" << "\n";
        linePrint("=",50);
        for (Counter d = 0; d < numDims; d++) {
          for (Side s = Left; s <= Right; ++s) {
            if (patch->getBoundaryType(d,s) == Patch::CoarseFine) {
              makeConnections(Matrix,hier,stencil,level,patch,d,s,FineToCoarse);
            } // end if boundary is CF interface
          } // end for s
        } // end for d
      }

      if (level < numLevels-1) {
        linePrint("=",50);
        dbg0 << "Building coarse-to-fine connections" 
             << " Patch ID = " << patch->_patchID
             << "\n";
        linePrint("=",50);
        //  const int numDims   = _param->numDims;
        const Vector<Counter>& refRat = hier._levels[level+1]->_refRat;
        dbg.indent();
        // List of fine patches covering this patch
        vector<Patch*> finePatchList = hier.finePatchesOverMe(*patch);
        Box coarseRefined(patch->_box.get(Left) * refRat,
                          (patch->_box.get(Right) + 1) * refRat - 1);
        dbg << "coarseRefined " << coarseRefined << "\n";
    
        //===================================================================
        // Loop over next-finer level patches that cover this patch
        //===================================================================
        for (vector<Patch*>::iterator iter = finePatchList.begin();
             iter != finePatchList.end(); ++iter) {
          //===================================================================
          // Compute relevant boxes at coarse and fine levels
          //===================================================================
          Patch* finePatch = *iter;
          dbg.setLevel(3);
          dbg << "Considering patch "
              << "ID=" << setw(2) << left << finePatch->_patchID << " "
              << "owner=" << setw(2) << left << finePatch->_procID << " "
              << finePatch->_box << " ..." << "\n";
          // Intersection of fine and coarse patches in fine-level subscripts
          Box fineIntersect = coarseRefined.intersect(finePatch->_box);
          // Intersection of fine and coarse patches in coarse-level subscripts
          Box coarseUnderFine(fineIntersect.get(Left) / refRat, 
                              fineIntersect.get(Right) / refRat);
          
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
            for (Side s = Left; s <= Right; ++s) {
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

void
HypreDriverSStruct::gatherSolutionVector(void)
{
  HypreDriverSStruct* sstructDriver =
    dynamic_cast<HypreDriverSStruct*>(hypreDriver);
  if (!sstructDriver) {
    throw InternalError("interface = SStruct but HypreDriver is not!",
                        __FILE__, __LINE__);
  }
  HYPRE_SStructVectorGather(sstructDriver->getX());
} // end HypreDriverStruct::gatherSolutionVector()

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

HypreDriver::Side& operator++(HypreDriver::Side &s)
{
  return s = Side(s+2);
}

std::ostream&
operator << (std::ostream& os, const HypreDriver::Side& s)
  // Write side s to the stream os.
{
  if      (s == Left ) os << "Left ";
  else if (s == Right) os << "Right";
  else os << "N/A";
  return os;
}

HypreDriver*
newHypreDriver(const HypreInterface& interface,,
               const Level* level,
               const MaterialSet* matlset,
               const VarLabel* A, Task::WhichDW which_A_dw,
               const VarLabel* x, bool modifies_x,
               const VarLabel* b, Task::WhichDW which_b_dw,
               const VarLabel* guess,
               Task::WhichDW which_guess_dw,
               const HypreSolverParams* params)
{
  switch (interface) {
  case HypreInterface::Struct: 
    {
      return new HypreDriverStruct(level.get_rep(), matls, A, which_A_dw,
                                   x, modifies_x, b, which_b_dw, guess, 
                                   which_guess_dw, dparams);
    }
  case HypreInterface::SStruct
    {
      return new HypreDriverSStruct(level.get_rep(), matls, A, which_A_dw,
                                    x, modifies_x, b, which_b_dw, guess, 
                                    which_guess_dw, dparams);
    }
  default:
    throw InternalError("Unsupported Hypre Interface: "+interface,
                        __FILE__, __LINE__);
  } // end switch (interface)
}

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
