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

#include <Packages/Uintah/CCA/Components/Solvers/HypreDriver.h>
#include <Packages/Uintah/CCA/Components/Solvers/HypreSolverParams.h>
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

// hypre includes
//#define HYPRE_TIMING
#ifndef HYPRE_TIMING
#ifndef hypre_ClearTiming
// This isn't in utilities.h for some reason...
#define hypre_ClearTiming()
#endif
#endif

using namespace Uintah;
//__________________________________
//  To turn on normal output
//  setenv SCI_DEBUG "HYPRE_DOING_COUT:+"

static DebugStream cout_doing("HYPRE_DOING_COUT", false);

namespace Uintah {

  /*_____________________________________________________________________
    class HypreInterface implementation common to all Types
    _____________________________________________________________________*/

    template<class Types>
      void HypreInterface::makeLinearSystem(const HypreInterface& interface)
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
    }


    template<class Types>
      void HypreInterface::getSolution(const HypreInterface& interface)
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
    }


  /*_____________________________________________________________________
    class HypreInterface implementation for CC variables, Struct interface
    _____________________________________________________________________*/

  void HypreInterface::makeLinearSystem<CCTypes>(const HypreInterface& interface)
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
  } // end HypreInterface<CCTypes>::makeLinearSystemStruct()


  void HypreInterface<CCTypes>::getSolutionStruct(void)
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
  } // end HypreInterface<CCTypes>::getSolutionStruct()


  /*_____________________________________________________________________
    class HypreInterface implementation for CC variables, SStruct interface
    _____________________________________________________________________*/

} // end namespace Uintah
