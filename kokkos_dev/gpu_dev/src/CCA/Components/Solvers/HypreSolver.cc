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


#include <CCA/Components/Solvers/HypreSolver.h>
#include <CCA/Components/Solvers/MatrixUtil.h>
#include <Core/Grid/DbgOutput.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/Stencil7.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Exceptions/ConvergenceFailure.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <CCA/Ports/Scheduler.h>
#include <CCA/Ports/LoadBalancer.h>
#include <Core/Containers/Array1.h>
#include <Core/Containers/Array2.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Math/MiscMath.h>
#include <Core/Math/MinMax.h>
#include <Core/Thread/Time.h>
#include <Core/Util/DebugStream.h>
#include <iomanip>

// hypre includes
//#define HYPRE_TIMING
#include <_hypre_utilities.h>
#include <HYPRE_struct_ls.h>
#include <_hypre_struct_mv.h>
#include <krylov.h>
#ifndef HYPRE_TIMING
  #ifndef hypre_ClearTiming
    // This isn't in utilities.h for some reason...
    #define hypre_ClearTiming()
  #endif
#endif

//#define PRINTSYSTEM

using std::cout;
using namespace Uintah;
//__________________________________
//  To turn on normal output
//  setenv SCI_DEBUG "HYPRE_DOING_COUT:+"

static DebugStream cout_doing("HYPRE_DOING_COUT", false);

namespace Uintah {

  HypreSolver2::HypreSolver2(const ProcessorGroup* myworld)
    : UintahParallelComponent(myworld)
  {
  }

  HypreSolver2::~HypreSolver2()
  {
  }

  class HypreSolver2Params : public SolverParameters {
  public:
    HypreSolver2Params()
    {
    }
    ~HypreSolver2Params() {}
    string solvertype;
    string precondtype;
    double tolerance;
    int maxiterations;
    int npre;
    int npost;
    int skip;
    int jump;
    int logging;
		int relax_type; 
    bool symmetric;
  };

  template<class Types>
  class HypreStencil7 : public RefCounted {
  public:
    HypreStencil7(const Level* level,
                  const MaterialSet* matlset,
                  const VarLabel* A, 
                  Task::WhichDW which_A_dw,
                  const VarLabel* x, 
                  bool modifies_x,
                  const VarLabel* b, 
                  Task::WhichDW which_b_dw,
                  const VarLabel* guess,
                  Task::WhichDW which_guess_dw,
                  const HypreSolver2Params* params)
      : level(level), matlset(matlset),
        A_label(A), which_A_dw(which_A_dw),
        X_label(x), 
        B_label(b), which_b_dw(which_b_dw),
        modifies_x(modifies_x),
        guess_label(guess), which_guess_dw(which_guess_dw), params(params)
    {
    }

    virtual ~HypreStencil7() {
    }

    //______________________________________________________________________
    void solve(const ProcessorGroup* pg, 
               const PatchSubset* patches,
               const MaterialSubset* matls,
               DataWarehouse* old_dw, 
               DataWarehouse* new_dw,
               Handle<HypreStencil7<Types> >)
    {
      typedef typename Types::sol_type sol_type;


      DataWarehouse* A_dw = new_dw->getOtherDataWarehouse(which_A_dw);
      DataWarehouse* b_dw = new_dw->getOtherDataWarehouse(which_b_dw);
      DataWarehouse* guess_dw = new_dw->getOtherDataWarehouse(which_guess_dw);
    
      ASSERTEQ(sizeof(Stencil7), 7*sizeof(double));
      double tstart = Time::currentSeconds();
      for(int m = 0;m<matls->size();m++){
        int matl = matls->get(m);

        //__________________________________
        // Setup grid
        HYPRE_StructGrid grid;
        HYPRE_StructGridCreate(pg->getComm(), 3, &grid);
        
        for(int p=0;p<patches->size();p++){
          const Patch* patch = patches->get(p);
          Patch::VariableBasis basis = Patch::translateTypeToBasis(sol_type::getTypeDescription()->getType(), true);

          IntVector l,h1;
          if(params->getSolveOnExtraCells()) {
            l  = patch->getExtraLowIndex(basis, IntVector(0,0,0));
            h1 = patch->getExtraHighIndex(basis, IntVector(0,0,0))-IntVector(1,1,1);
          } else {
            l = patch->getLowIndex(basis);
            h1 = patch->getHighIndex(basis)-IntVector(1,1,1);
          }
          
          HYPRE_StructGridSetExtents(grid, l.get_pointer(), h1.get_pointer());
        }
        
        // Periodic boundaries
        const Level* level = getLevel(patches);
        IntVector periodic_vector = level->getPeriodicBoundaries();
        
        IntVector low, high;
        level->findCellIndexRange(low, high);
        IntVector range = high-low;
        
        int periodic[3];
        periodic[0] = periodic_vector.x() * range.x();
        periodic[1] = periodic_vector.y() * range.y();
        periodic[2] = periodic_vector.z() * range.z();
        HYPRE_StructGridSetPeriodic(grid, periodic);
        
        // Assemble the grid
        HYPRE_StructGridAssemble(grid);

        //__________________________________
        // Create the stencil
        HYPRE_StructStencil stencil;
        if(params->symmetric){
          
          HYPRE_StructStencilCreate(3, 4, &stencil);
          int offsets[4][3] = {{0,0,0},
                               {-1,0,0},
                               {0,-1,0},
                               {0,0,-1}};
          for(int i=0;i<4;i++) {
            HYPRE_StructStencilSetElement(stencil, i, offsets[i]);
          }
          
        } else {
          
          HYPRE_StructStencilCreate(3, 7, &stencil);
          int offsets[7][3] = {{0,0,0},
                               {-1,0,0}, {1,0,0},
                               {0,1,0}, {0,-1,0},
                               {0,0,1}, {0,0,-1}};
                               
          for(int i=0;i<7;i++){
            HYPRE_StructStencilSetElement(stencil, i, offsets[i]);
          }
        }

        //__________________________________
        // Create the matrix
        HYPRE_StructMatrix HA;
        HYPRE_StructMatrixCreate(pg->getComm(), grid, stencil, &HA);
        HYPRE_StructMatrixSetSymmetric(HA, params->symmetric);
        int ghost[] = {1,1,1,1,1,1};
        HYPRE_StructMatrixSetNumGhost(HA, ghost);
        HYPRE_StructMatrixInitialize(HA);

        for(int p=0;p<patches->size();p++){
          const Patch* patch = patches->get(p);
          printTask( patches, patch, cout_doing, "HypreSolver:solve: Create Matrix" );
          //__________________________________
          // Get A matrix from the DW
          typename Types::matrix_type A;
          A_dw->get(A, A_label, matl, patch, Ghost::None, 0);

          Patch::VariableBasis basis = Patch::translateTypeToBasis(sol_type::getTypeDescription()->getType(), true);

          IntVector l,h;
          if(params->getSolveOnExtraCells()){
            l = patch->getExtraLowIndex(basis, IntVector(0,0,0));
            h = patch->getExtraHighIndex(basis, IntVector(0,0,0));
          } else {
            l = patch->getLowIndex(basis);
            h = patch->getHighIndex(basis);
          }

          //__________________________________
          // Feed it to Hypre
          if(params->symmetric){
            double* values = scinew double[(h.x()-l.x())*4]; 
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
                                               ll.get_pointer(), hh.get_pointer(),
                                               4, stencil_indices, values);

              } // y loop
            }  // z loop
            delete[] values;
          } else {
            int stencil_indices[] = {0,1,2,3,4,5,6};
            
            for(int z=l.z();z<h.z();z++){
              for(int y=l.y();y<h.y();y++){
            
                const double* values = &A[IntVector(l.x(), y, z)].p;
                IntVector ll(l.x(), y, z);
                IntVector hh(h.x()-1, y, z);
                HYPRE_StructMatrixSetBoxValues(HA,
                                               ll.get_pointer(), hh.get_pointer(),
                                               7, stencil_indices,
                                               const_cast<double*>(values));
              }  // y loop
            } // z loop
          }
        }
        HYPRE_StructMatrixAssemble(HA);

        //__________________________________
        // Create the RHS
        HYPRE_StructVector HB;
        HYPRE_StructVectorCreate(pg->getComm(), grid, &HB);
        HYPRE_StructVectorInitialize(HB);

        for(int p=0;p<patches->size();p++){
          const Patch* patch = patches->get(p);
          printTask( patches, patch, cout_doing, "HypreSolver:solve: Create RHS" );

          //__________________________________
          // Get the B vector from the DW
          typename Types::const_type B;
          b_dw->get(B, B_label, matl, patch, Ghost::None, 0);

          Patch::VariableBasis basis = Patch::translateTypeToBasis(sol_type::getTypeDescription()->getType(), true);

          IntVector l,h;
          if(params->getSolveOnExtraCells()) {
            l = patch->getExtraLowIndex(basis,  IntVector(0,0,0));
            h = patch->getExtraHighIndex(basis, IntVector(0,0,0));
          } else {
            l = patch->getLowIndex(basis);
            h = patch->getHighIndex(basis);
          }

          //__________________________________
          // Feed it to Hypre
          for(int z=l.z();z<h.z();z++){
            for(int y=l.y();y<h.y();y++){
              const double* values = &B[IntVector(l.x(), y, z)];
              IntVector ll(l.x(), y, z);
              IntVector hh(h.x()-1, y, z);
              HYPRE_StructVectorSetBoxValues(HB,
                                             ll.get_pointer(), hh.get_pointer(),
                                             const_cast<double*>(values));
            }
          }
        }
        HYPRE_StructVectorAssemble(HB);

        //__________________________________
        // Create the solution vector
        HYPRE_StructVector HX;
        HYPRE_StructVectorCreate(pg->getComm(), grid, &HX);
        HYPRE_StructVectorInitialize(HX);

        for(int p=0;p<patches->size();p++){
          const Patch* patch = patches->get(p);
          printTask( patches, patch, cout_doing, "HypreSolver:solve: Create X" );
          
          //__________________________________
          // Get the initial guess
          if(guess_label){
            typename Types::const_type X;
            guess_dw->get(X, guess_label, matl, patch, Ghost::None, 0);

            Patch::VariableBasis basis = Patch::translateTypeToBasis(sol_type::getTypeDescription()->getType(), true);

            IntVector l,h;
            if(params->getSolveOnExtraCells()){
              l = patch->getExtraLowIndex(basis, IntVector(0,0,0));
              h = patch->getExtraHighIndex(basis, IntVector(0,0,0));
            }
            else{
              l = patch->getLowIndex(basis);
              h = patch->getHighIndex(basis);
            }

            //__________________________________
            // Feed it to Hypre
            for(int z=l.z();z<h.z();z++){
              for(int y=l.y();y<h.y();y++){
                const double* values = &X[IntVector(l.x(), y, z)];
                IntVector ll(l.x(), y, z);
                IntVector hh(h.x()-1, y, z);
                HYPRE_StructVectorSetBoxValues(HX,
                                               ll.get_pointer(), hh.get_pointer(),
                                               const_cast<double*>(values));
              }
            }
          }  // initialGuess
        } // patch loop
        HYPRE_StructVectorAssemble(HX);
        
        //__________________________________
        //  Dynamic tolerances  Arches uses this
        double precond_tolerance = 0.0;
        
        double solve_start = Time::currentSeconds();
        int num_iterations;
        double final_res_norm;
        
        //______________________________________________________________________
        // Solve the system
        if (params->solvertype == "SMG" || params->solvertype == "smg"){
          int time_index = hypre_InitializeTiming("SMG Setup");
          hypre_BeginTiming(time_index);

          HYPRE_StructSolver  solver;
          HYPRE_StructSMGCreate         (MPI_COMM_WORLD, &solver);         
          HYPRE_StructSMGSetMemoryUse   (solver,  0);                      
          HYPRE_StructSMGSetMaxIter     (solver,  params->maxiterations);  
          HYPRE_StructSMGSetTol         (solver,  params->tolerance);      
          HYPRE_StructSMGSetRelChange   (solver,  0);                      
          HYPRE_StructSMGSetNumPreRelax (solver,  params->npre);           
          HYPRE_StructSMGSetNumPostRelax(solver,  params->npost);
          HYPRE_StructSMGSetLogging     (solver,  params->logging);        
          HYPRE_StructSMGSetup          (solver,  HA, HB, HX);           


          hypre_EndTiming(time_index);
          hypre_PrintTiming("Setup phase times", pg->getComm());
          hypre_FinalizeTiming(time_index);
          hypre_ClearTiming();

          time_index = hypre_InitializeTiming("SMG Solve");
          hypre_BeginTiming(time_index);

          HYPRE_StructSMGSolve(solver, HA, HB, HX);

          hypre_EndTiming(time_index);
          hypre_PrintTiming("Solve phase times", pg->getComm());
          hypre_FinalizeTiming(time_index);
          hypre_ClearTiming();
   
          HYPRE_StructSMGGetNumIterations(solver, &num_iterations);
          HYPRE_StructSMGGetFinalRelativeResidualNorm(solver, &final_res_norm);
          HYPRE_StructSMGDestroy(solver);
        } else if(params->solvertype == "PFMG" || params->solvertype == "pfmg"){

          int time_index = hypre_InitializeTiming("PFMG Setup");
          hypre_BeginTiming(time_index);

          HYPRE_StructSolver  solver;
          HYPRE_StructPFMGCreate        (pg->getComm(),&solver);
          HYPRE_StructPFMGSetMaxIter    (solver,      params->maxiterations);
          HYPRE_StructPFMGSetTol        (solver,      params->tolerance);
          HYPRE_StructPFMGSetRelChange  (solver,      0);

          HYPRE_StructPFMGSetRelaxType   (solver,  params->relax_type);                           
          HYPRE_StructPFMGSetNumPreRelax (solver,  params->npre);                
          HYPRE_StructPFMGSetNumPostRelax(solver,  params->npost);               
          HYPRE_StructPFMGSetSkipRelax   (solver,  params->skip);                
          HYPRE_StructPFMGSetLogging     (solver,  params->logging);             
          HYPRE_StructPFMGSetup          (solver,  HA, HB,  HX);  


          hypre_EndTiming     (time_index);
          hypre_PrintTiming   ("Setup phase times", pg->getComm());
          hypre_FinalizeTiming(time_index);
          hypre_ClearTiming();

          time_index = hypre_InitializeTiming("PFMG Solve");
          hypre_BeginTiming(time_index);

          HYPRE_StructPFMGSolve(solver, HA, HB, HX);
        
          hypre_EndTiming     (time_index);
          hypre_PrintTiming   ("Solve phase times", pg->getComm());
          hypre_FinalizeTiming(time_index);
          hypre_ClearTiming();
      
          HYPRE_StructPFMGGetNumIterations            (solver, &num_iterations);
          HYPRE_StructPFMGGetFinalRelativeResidualNorm(solver, &final_res_norm);
          HYPRE_StructPFMGDestroy(solver);

        } else if(params->solvertype == "SparseMSG" || params->solvertype == "sparsemsg"){
          int time_index = hypre_InitializeTiming("SparseMSG Setup");
          hypre_BeginTiming(time_index);

          HYPRE_StructSolver  solver;
          HYPRE_StructSparseMSGCreate      (pg->getComm(), &solver);              
          HYPRE_StructSparseMSGSetMaxIter  (solver, params->maxiterations);       
          HYPRE_StructSparseMSGSetJump     (solver, params->jump);                
          HYPRE_StructSparseMSGSetTol      (solver, params->tolerance);           
          HYPRE_StructSparseMSGSetRelChange(solver, 0);                           

          HYPRE_StructSparseMSGSetRelaxType   (solver,  params->relax_type);                           
          HYPRE_StructSparseMSGSetNumPreRelax (solver,  params->npre);                
          HYPRE_StructSparseMSGSetNumPostRelax(solver,  params->npost);               
          HYPRE_StructSparseMSGSetLogging     (solver,  params->logging);             
          HYPRE_StructSparseMSGSetup          (solver,  HA, HB,  HX);  


          hypre_EndTiming     (time_index);
          hypre_PrintTiming   ("Setup phase times", pg->getComm());
          hypre_FinalizeTiming(time_index);
          hypre_ClearTiming();

          time_index = hypre_InitializeTiming("SparseMSG Solve");
          hypre_BeginTiming(time_index);

          HYPRE_StructSparseMSGSolve(solver, HA, HB, HX);

          hypre_EndTiming(time_index);
          hypre_PrintTiming("Solve phase times", pg->getComm());
          hypre_FinalizeTiming(time_index);
          hypre_ClearTiming();
   
          HYPRE_StructSparseMSGGetNumIterations            (solver, &num_iterations);
          HYPRE_StructSparseMSGGetFinalRelativeResidualNorm(solver, &final_res_norm);
          HYPRE_StructSparseMSGDestroy(solver);
          //__________________________________
          //
        } else if(params->solvertype == "CG" || params->solvertype == "cg" || params->solvertype == "conjugategradient" || params->solvertype == "PCG" || params->solvertype == "cg"){
          int time_index = hypre_InitializeTiming("PCG Setup");
          hypre_BeginTiming(time_index);

          HYPRE_StructSolver solver;
          HYPRE_StructPCGCreate(pg->getComm(), &solver);                
          HYPRE_PCGSetMaxIter  ((HYPRE_Solver)  solver,  params->maxiterations);  
          HYPRE_PCGSetTol      ((HYPRE_Solver)  solver,  params->tolerance);      
          HYPRE_PCGSetTwoNorm  ((HYPRE_Solver)  solver,  1);                      
          HYPRE_PCGSetRelChange((HYPRE_Solver)  solver,  0);                      
          HYPRE_PCGSetLogging  ((HYPRE_Solver)  solver,  params->logging);        

          HYPRE_PtrToSolverFcn precond;
          HYPRE_PtrToSolverFcn precond_setup;
          HYPRE_StructSolver precond_solver;
          setupPrecond(pg, precond, precond_setup, precond_solver, precond_tolerance);
          
          HYPRE_PCGSetPrecond((HYPRE_Solver)solver, 
                               precond,
                               precond_setup, 
                              (HYPRE_Solver)precond_solver);
                              
          HYPRE_PCGSetup( (HYPRE_Solver)solver, 
                          (HYPRE_Matrix)HA,
                          (HYPRE_Vector)HB, 
                          (HYPRE_Vector)HX);
                          
          hypre_EndTiming     (time_index);
          hypre_PrintTiming   ("Setup phase times", pg->getComm());
          hypre_FinalizeTiming(time_index);
          hypre_ClearTiming();
   
          time_index = hypre_InitializeTiming("PCG Solve");
          hypre_BeginTiming(time_index);

          HYPRE_PCGSolve( (HYPRE_Solver)solver, 
                          (HYPRE_Matrix)HA, 
                          (HYPRE_Vector)HB, 
                          (HYPRE_Vector)HX);

          hypre_EndTiming     (time_index);
          hypre_PrintTiming   ("Solve phase times", pg->getComm());
          hypre_FinalizeTiming(time_index);
          hypre_ClearTiming();

          HYPRE_PCGGetNumIterations            ( (HYPRE_Solver)solver, &num_iterations );
          HYPRE_PCGGetFinalRelativeResidualNorm( (HYPRE_Solver)solver, &final_res_norm );
          HYPRE_StructPCGDestroy(solver);

          destroyPrecond(precond_solver);
        } else if(params->solvertype == "Hybrid" || params->solvertype == "hybrid"){
          /*-----------------------------------------------------------
           * Solve the system using Hybrid
           *-----------------------------------------------------------*/

          int time_index = hypre_InitializeTiming("Hybrid Setup");
          hypre_BeginTiming(time_index);

          HYPRE_StructSolver  solver;
          HYPRE_StructHybridCreate           (pg->getComm(), &solver);  
          HYPRE_StructHybridSetDSCGMaxIter   (solver, 100);                                    
          HYPRE_StructHybridSetPCGMaxIter    (solver, params->maxiterations);                  
          HYPRE_StructHybridSetTol           (solver, params->tolerance);                      
          HYPRE_StructHybridSetConvergenceTol(solver, 0.90);                                   
          HYPRE_StructHybridSetTwoNorm       (solver, 1);                                      
          HYPRE_StructHybridSetRelChange     (solver, 0);                                      
          HYPRE_StructHybridSetLogging       (solver, params->logging);                        


          HYPRE_PtrToSolverFcn precond;
          HYPRE_PtrToSolverFcn precond_setup;
          HYPRE_StructSolver   precond_solver;
          setupPrecond(pg, precond, precond_setup, precond_solver, precond_tolerance);
          HYPRE_StructHybridSetPrecond(solver,
                                       (HYPRE_PtrToStructSolverFcn)precond,
                                       (HYPRE_PtrToStructSolverFcn)precond_setup,
                                       (HYPRE_StructSolver)precond_solver);

          HYPRE_StructHybridSetup(solver, HA, HB, HX);

          hypre_EndTiming     (time_index);
          hypre_PrintTiming   ("Setup phase times", pg->getComm());
          hypre_FinalizeTiming(time_index);
          hypre_ClearTiming();
   
          time_index = hypre_InitializeTiming("Hybrid Solve");
          hypre_BeginTiming(time_index);

          HYPRE_StructHybridSolve(solver, HA, HB, HX);

          hypre_EndTiming     (time_index);
          hypre_PrintTiming   ("Solve phase times", pg->getComm());
          hypre_FinalizeTiming(time_index);
          hypre_ClearTiming();

          HYPRE_StructHybridGetNumIterations            (solver, &num_iterations);
          HYPRE_StructHybridGetFinalRelativeResidualNorm(solver, &final_res_norm);
          HYPRE_StructHybridDestroy(solver);

          destroyPrecond(precond_solver);
          //__________________________________
          //
        } else if(params->solvertype == "GMRES" || params->solvertype == "gmres"){
          int time_index = hypre_InitializeTiming("GMRES Setup");
          hypre_BeginTiming(time_index);

          HYPRE_StructSolver solver;
          HYPRE_StructGMRESCreate(pg->getComm(),  &solver);                
          HYPRE_GMRESSetMaxIter  ((HYPRE_Solver)  solver,  params->maxiterations);  
          HYPRE_GMRESSetTol      ((HYPRE_Solver)  solver,  params->tolerance);      
          HYPRE_GMRESSetRelChange((HYPRE_Solver)  solver,  0);                      
          HYPRE_GMRESSetLogging  ((HYPRE_Solver)  solver,  params->logging);        

          
          HYPRE_PtrToSolverFcn precond;
          HYPRE_PtrToSolverFcn precond_setup;
          HYPRE_StructSolver   precond_solver;
          
          setupPrecond(pg, precond, precond_setup, precond_solver, precond_tolerance);
          
          HYPRE_GMRESSetPrecond((HYPRE_Solver)solver, precond, precond_setup,
                                (HYPRE_Solver)precond_solver);
          
          HYPRE_GMRESSetup( (HYPRE_Solver)solver, 
                            (HYPRE_Matrix)HA, 
                            (HYPRE_Vector)HB, 
                            (HYPRE_Vector)HX);

          hypre_EndTiming     (time_index);
          hypre_PrintTiming   ("Setup phase times", pg->getComm());
          hypre_FinalizeTiming(time_index);
          hypre_ClearTiming();
   
          time_index = hypre_InitializeTiming("GMRES Solve");
          hypre_BeginTiming(time_index);

          HYPRE_GMRESSolve( (HYPRE_Solver)solver, 
                            (HYPRE_Matrix)HA, 
                            (HYPRE_Vector)HB, 
                            (HYPRE_Vector)HX);

          hypre_EndTiming     (time_index);
          hypre_PrintTiming   ("Solve phase times", pg->getComm());
          hypre_FinalizeTiming(time_index);
          hypre_ClearTiming();

          HYPRE_GMRESGetNumIterations            ( (HYPRE_Solver)solver, &num_iterations);
          HYPRE_GMRESGetFinalRelativeResidualNorm( (HYPRE_Solver)solver, &final_res_norm);
          HYPRE_StructGMRESDestroy(solver);
        
          destroyPrecond(precond_solver);
        } else {
          throw InternalError("Unknown solver type: "+params->solvertype, __FILE__, __LINE__);
        }
        
        
#ifdef PRINTSYSTEM
        //__________________________________
        //   Debugging 
        vector<string> fname;   
        params->getOutputFileName(fname);
        HYPRE_StructMatrixPrint(fname[0].c_str(), HA, 0);
        HYPRE_StructVectorPrint(fname[1].c_str(), HB, 0);
        HYPRE_StructVectorPrint(fname[2].c_str(), HX, 0);
#endif
        
        printTask( patches, patches->get(0), cout_doing, "HypreSolver:solve: testConvergence" );
        //__________________________________
        // Test for convergence
        if(final_res_norm > params->tolerance || finite(final_res_norm) == 0){
          if( params->getRestartTimestepOnFailure() ){
            if(pg->myrank() == 0)
              cout << "HypreSolver not converged in " << num_iterations 
                   << "iterations, final residual= " << final_res_norm 
                   << ", requesting smaller timestep\n";
            //new_dw->abortTimestep();
            //new_dw->restartTimestep();
          } else {
            throw ConvergenceFailure("HypreSolver variable: "+X_label->getName()+", solver: "+params->solvertype+", preconditioner: "+params->precondtype,
                                     num_iterations, final_res_norm,
                                     params->tolerance,__FILE__,__LINE__);
          }
        }
        //__________________________________
        // Push the solution into Uintah data structure
        double solve_dt = Time::currentSeconds()-solve_start;

        for(int p=0;p<patches->size();p++){
          const Patch* patch = patches->get(p);
          printTask( patches, patch, cout_doing, "HypreSolver:solve: copy solution" );
          Patch::VariableBasis basis = Patch::translateTypeToBasis(sol_type::getTypeDescription()->getType(), true);

          IntVector l,h;
          if(params->getSolveOnExtraCells()){
            l = patch->getExtraLowIndex(basis,  IntVector(0,0,0));
            h = patch->getExtraHighIndex(basis, IntVector(0,0,0));
          }else{
            l = patch->getLowIndex(basis);
            h = patch->getHighIndex(basis);
          }
          CellIterator iter(l, h);

          typename Types::sol_type Xnew;
          if(modifies_x){
            new_dw->getModifiable(Xnew, X_label, matl, patch);
          }else{
            new_dw->allocateAndPut(Xnew, X_label, matl, patch);
          }

          // Get the solution back from hypre
          for(int z=l.z();z<h.z();z++){
            for(int y=l.y();y<h.y();y++){
              const double* values = &Xnew[IntVector(l.x(), y, z)];
              IntVector ll(l.x(), y, z);
              IntVector hh(h.x()-1, y, z);
              HYPRE_StructVectorGetBoxValues(HX,
                  ll.get_pointer(), hh.get_pointer(),
                  const_cast<double*>(values));
            }
          }       
        }
        //__________________________________
        // clean up
        HYPRE_StructMatrixDestroy(HA);
        HYPRE_StructVectorDestroy(HB);
        HYPRE_StructVectorDestroy(HX);
        HYPRE_StructStencilDestroy(stencil);
        HYPRE_StructGridDestroy(grid);

        double dt=Time::currentSeconds()-tstart;
        if(pg->myrank() == 0){
          cout << "Solve of " << X_label->getName() 
               << " on level " << level->getIndex()
               << " completed in " << dt 
               << " seconds (solve only: " << solve_dt 
               << " seconds, " << num_iterations 
               << " iterations, residual=" << final_res_norm << ")\n";
        }
        tstart = Time::currentSeconds();
      }  // matl loop
    }
    
    //______________________________________________________________________
    void setupPrecond(const ProcessorGroup* pg,
                      HYPRE_PtrToSolverFcn& precond,
                      HYPRE_PtrToSolverFcn& pcsetup,
                      HYPRE_StructSolver& precond_solver,
                      const double precond_tolerance){
                      
      if(params->precondtype == "SMG" || params->precondtype == "smg"){
        /* use symmetric SMG as preconditioner */
        HYPRE_StructSMGCreate         (pg->getComm(),    &precond_solver);  
        HYPRE_StructSMGSetMemoryUse   (precond_solver,   0);                                 
        HYPRE_StructSMGSetMaxIter     (precond_solver,   1);                                 
        HYPRE_StructSMGSetTol         (precond_solver,   precond_tolerance);                               
        HYPRE_StructSMGSetZeroGuess   (precond_solver);                                      
        HYPRE_StructSMGSetNumPreRelax (precond_solver,   params->npre);                      
        HYPRE_StructSMGSetNumPostRelax(precond_solver,   params->npost);                     
        HYPRE_StructSMGSetLogging     (precond_solver,   0);                                 

        
        precond = (HYPRE_PtrToSolverFcn)HYPRE_StructSMGSolve;
        pcsetup = (HYPRE_PtrToSolverFcn)HYPRE_StructSMGSetup;
      //__________________________________
      //
      } else if(params->precondtype == "PFMG" || params->precondtype == "pfmg"){
        /* use symmetric PFMG as preconditioner */
        HYPRE_StructPFMGCreate        (pg->getComm(),    &precond_solver);
        HYPRE_StructPFMGSetMaxIter    (precond_solver,   1);
        HYPRE_StructPFMGSetTol        (precond_solver,   precond_tolerance); 
        HYPRE_StructPFMGSetZeroGuess  (precond_solver);

        HYPRE_StructPFMGSetRelaxType   (precond_solver,  params->relax_type);              
        HYPRE_StructPFMGSetNumPreRelax (precond_solver,  params->npre);   
        HYPRE_StructPFMGSetNumPostRelax(precond_solver,  params->npost);  
        HYPRE_StructPFMGSetSkipRelax   (precond_solver,  params->skip);   
        HYPRE_StructPFMGSetLogging     (precond_solver,  0);              

        precond = (HYPRE_PtrToSolverFcn)HYPRE_StructPFMGSolve;
        pcsetup = (HYPRE_PtrToSolverFcn)HYPRE_StructPFMGSetup;
      //__________________________________
      //
      } else if(params->precondtype == "SparseMSG" || params->precondtype == "sparsemsg"){
        /* use symmetric SparseMSG as preconditioner */
        HYPRE_StructSparseMSGCreate       (pg->getComm(),   &precond_solver);                  
        HYPRE_StructSparseMSGSetMaxIter   (precond_solver,  1);                                
        HYPRE_StructSparseMSGSetJump      (precond_solver,  params->jump);                     
        HYPRE_StructSparseMSGSetTol       (precond_solver,  precond_tolerance);                              
        HYPRE_StructSparseMSGSetZeroGuess (precond_solver);                                    

        HYPRE_StructSparseMSGSetRelaxType   (precond_solver,  params->relax_type);              
        HYPRE_StructSparseMSGSetNumPreRelax (precond_solver,  params->npre);   
        HYPRE_StructSparseMSGSetNumPostRelax(precond_solver,  params->npost);  
        HYPRE_StructSparseMSGSetLogging     (precond_solver,  0);              
        
        precond = (HYPRE_PtrToSolverFcn)HYPRE_StructSparseMSGSolve;
        pcsetup = (HYPRE_PtrToSolverFcn)HYPRE_StructSparseMSGSetup;
      //__________________________________
      //
      } else if(params->precondtype == "Jacobi" || params->precondtype == "jacobi"){
        /* use two-step Jacobi as preconditioner */
        HYPRE_StructJacobiCreate      (pg->getComm(),    &precond_solver);  
        HYPRE_StructJacobiSetMaxIter  (precond_solver,   2);                       
        HYPRE_StructJacobiSetTol      (precond_solver,   precond_tolerance);                     
        HYPRE_StructJacobiSetZeroGuess(precond_solver);                            

        
        precond = (HYPRE_PtrToSolverFcn)HYPRE_StructJacobiSolve;
        pcsetup = (HYPRE_PtrToSolverFcn)HYPRE_StructJacobiSetup;
      //__________________________________
      //
      } else if(params->precondtype == "Diagonal" || params->precondtype == "diagonal"){
        /* use diagonal scaling as preconditioner */
#ifdef HYPRE_USE_PTHREADS
        for (i = 0; i < hypre_NumThreads; i++)
          precond[i] = NULL;
#else
        precond = NULL;
#endif
        precond = (HYPRE_PtrToSolverFcn)HYPRE_StructDiagScale;
        pcsetup = (HYPRE_PtrToSolverFcn)HYPRE_StructDiagScaleSetup;
      } else {
        // This should have been caught in readParameters...
        throw InternalError("Unknown preconditionertype: "+params->precondtype, __FILE__, __LINE__);
      }
    }
    void destroyPrecond(HYPRE_StructSolver precond_solver){
      if(params->precondtype        == "SMG"       || params->precondtype == "smg"){
        HYPRE_StructSMGDestroy(precond_solver);
      } else if(params->precondtype == "PFMG"      || params->precondtype == "pfmg"){
        HYPRE_StructPFMGDestroy(precond_solver);
      } else if(params->precondtype == "SparseMSG" || params->precondtype == "sparsemsg"){
        HYPRE_StructSparseMSGDestroy(precond_solver);
      } else if(params->precondtype == "Jacobi"    || params->precondtype == "jacobi"){
        HYPRE_StructJacobiDestroy(precond_solver);
      } else if(params->precondtype == "Diagonal"  || params->precondtype == "diagonal"){
      } else {
        // This should have been caught in readParameters...
        throw InternalError("Unknown preconditionertype in destroyPrecond: "+params->precondtype, __FILE__, __LINE__);
      }
    }
  //______________________________________________________________________
  private:

    const Level*       level;
    const MaterialSet* matlset;
    const VarLabel*    A_label;
    Task::WhichDW      which_A_dw;
    const VarLabel*    X_label;
    const VarLabel*    B_label;
    Task::WhichDW      which_b_dw;
    bool               modifies_x;
    const VarLabel*    guess_label;
    Task::WhichDW      which_guess_dw;
    const HypreSolver2Params* params;
  };
  
  //______________________________________________________________________
  SolverParameters* HypreSolver2::readParameters(ProblemSpecP& params, const string& varname)
  {
    HypreSolver2Params* p = new HypreSolver2Params();
    bool found=false;
    if(params){
      for(ProblemSpecP param = params->findBlock("Parameters"); param != 0;
          param = param->findNextBlock("Parameters")) {
        string variable;
        if(param->getAttribute("variable", variable) && variable != varname)
          continue;
          
        param->getWithDefault ("solver",          p->solvertype,     "smg");      
        param->getWithDefault ("preconditioner",  p->precondtype,    "diagonal"); 
        param->getWithDefault ("tolerance",       p->tolerance,      1.e-10);     
        param->getWithDefault ("maxiterations",   p->maxiterations,  75);         
        param->getWithDefault ("npre",            p->npre,           1);          
        param->getWithDefault ("npost",           p->npost,          1);          
        param->getWithDefault ("skip",            p->skip,           0);          
        param->getWithDefault ("jump",            p->jump,           0);          
        param->getWithDefault ("logging",         p->logging,        0);          
				param->getWithDefault ("relax_type",      p->relax_type,     1); // Jacobi = 0; weighted Jacobi = 1; 
																																			   // red-black GS symmetric = 2; red-black GS non-symmetrix = 3;

        found=true;
      }
    }
    if(!found){
      p->solvertype    = "smg";
      p->precondtype   = "diagonal";
      p->tolerance     = 1.e-10;
      p->maxiterations = 75;
      p->npre    = 1;
      p->npost   = 1;
      p->skip    = 0;
      p->jump    = 0;
      p->logging = 0;
			p->relax_type = 1; 
    }
    p->symmetric = true;
    return p;
  }
  
  //______________________________________________________________________
  void HypreSolver2::scheduleSolve(const LevelP& level, SchedulerP& sched,
                                   const MaterialSet* matls,
                                   const VarLabel* A,    Task::WhichDW which_A_dw,  
                                   const VarLabel* x,
                                   bool modifies_x,
                                   const VarLabel* b,    Task::WhichDW which_b_dw,  
                                   const VarLabel* guess,Task::WhichDW which_guess_dw,
                                   const SolverParameters* params)
  {
    printSchedule(level, cout_doing, "HypreSolver:scheduleSolve");
    Task* task;
    // The extra handle arg ensures that the stencil7 object will get freed
    // when the task gets freed.  The downside is that the refcount gets
    // tweaked everytime solve is called.

    TypeDescription::Type domtype = A->typeDescription()->getType();
    ASSERTEQ(domtype, x->typeDescription()->getType());
    ASSERTEQ(domtype, b->typeDescription()->getType());

    //__________________________________
    // bulletproofing
    IntVector periodic = level->getPeriodicBoundaries();
    if(periodic != IntVector(0,0,0)){
      IntVector l,h;
      level->findCellIndexRange( l, h );
      IntVector range = h - l;
      if( fmodf(range.x(),2) !=0 || fmodf(range.y(),2) != 0 || fmodf(range.z(),2) != 0) {
        ostringstream warn;
        warn << "\nINPUT FILE ERROR: hypre solver: \n"
             << "With periodic boundary conditions the resolution of your grid "<<range<<", in each direction, must be a power of 2 (i.e. 2^n), \n"
             << "e.g., 16,32,64,128....";
            
        throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
      }
    }
    
    const HypreSolver2Params* dparams = dynamic_cast<const HypreSolver2Params*>(params);
    if(!dparams){
      throw InternalError("Wrong type of params passed to hypre solver!", __FILE__, __LINE__);
    }

    switch(domtype){
    case TypeDescription::SFCXVariable:
      {
        HypreStencil7<SFCXTypes>* that = scinew HypreStencil7<SFCXTypes>(level.get_rep(), matls, A, which_A_dw, x, modifies_x, b, which_b_dw, guess, which_guess_dw, dparams);
        Handle<HypreStencil7<SFCXTypes> > handle = that;
        task = scinew Task("Hypre:Matrix solve (SFCX)", that, &HypreStencil7<SFCXTypes>::solve, handle);
      }
      break;
    case TypeDescription::SFCYVariable:
      {
        HypreStencil7<SFCYTypes>* that = scinew HypreStencil7<SFCYTypes>(level.get_rep(), matls, A, which_A_dw, x, modifies_x, b, which_b_dw, guess, which_guess_dw, dparams);
        Handle<HypreStencil7<SFCYTypes> > handle = that;
        task = scinew Task("Hypre:Matrix solve (SFCY)", that, &HypreStencil7<SFCYTypes>::solve, handle);
      }
      break;
    case TypeDescription::SFCZVariable:
      {
        HypreStencil7<SFCZTypes>* that = scinew HypreStencil7<SFCZTypes>(level.get_rep(), matls, A, which_A_dw, x, modifies_x, b, which_b_dw, guess, which_guess_dw, dparams);
        Handle<HypreStencil7<SFCZTypes> > handle = that;
        task = scinew Task("Hypre:Matrix solve (SFCZ)", that, &HypreStencil7<SFCZTypes>::solve, handle);
      }
      break;
    case TypeDescription::CCVariable:
      {
        HypreStencil7<CCTypes>* that = scinew HypreStencil7<CCTypes>(level.get_rep(), matls, A, which_A_dw, x, modifies_x, b, which_b_dw, guess, which_guess_dw, dparams);
        Handle<HypreStencil7<CCTypes> > handle = that;
        task = scinew Task("Hypre:Matrix solve (CC)", that, &HypreStencil7<CCTypes>::solve, handle);
      }
      break;
    case TypeDescription::NCVariable:
      {
        HypreStencil7<NCTypes>* that = scinew HypreStencil7<NCTypes>(level.get_rep(), matls, A, which_A_dw, x, modifies_x, b, which_b_dw, guess, which_guess_dw, dparams);
        Handle<HypreStencil7<NCTypes> > handle = that;
        task = scinew Task("Hypre:Matrix solve (NC)", that, &HypreStencil7<NCTypes>::solve, handle);
      }
      break;
    default:
      throw InternalError("Unknown variable type in scheduleSolve", __FILE__, __LINE__);
    }

    task->requires(which_A_dw, A, Ghost::None, 0);
    if(modifies_x)
      task->modifies(x);
    else
      task->computes(x);
    
    if(guess){
      task->requires(which_guess_dw, guess, Ghost::None, 0); 
    }

    task->requires(which_b_dw, b, Ghost::None, 0);
    LoadBalancer* lb = sched->getLoadBalancer();
    
    sched->addTask(task, lb->getPerProcessorPatchSet(level), matls);
  }
  

string HypreSolver2::getName(){
  return "hypre";
}

} // end namespace Uintah
