//--------------------------------------------------------------------------
// File: HypreDriverStruct.cc
//
// Implementation of the interface from Uintah to Hypre's Struct system
// interface, for cell-centered variables (e.g., pressure in implicit ICE).
//--------------------------------------------------------------------------
#include <Packages/Uintah/CCA/Components/Solvers/HypreSolverParams.h>
#include <Packages/Uintah/CCA/Components/Solvers/HypreDriverStruct.h>
#include <Packages/Uintah/CCA/Components/Solvers/HypreSolverBase.h>
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

//#####################################################################
// class HypreDriver implementation common to all variable types
//#####################################################################
  
HypreDriverStruct::~HypreDriverStruct(void)
  //___________________________________________________________________
  // HypreDriverStruct destructor
  //___________________________________________________________________
{
  HYPRE_StructMatrixDestroy(_HA);
  HYPRE_StructVectorDestroy(_HB);
  HYPRE_StructVectorDestroy(_HX);
  HYPRE_StructStencilDestroy(_stencil);
  HYPRE_StructGridDestroy(_grid);
}

void
HypreDriverStruct::printMatrix(const string& fileName /* =  "output" */)
{
  cout_doing << "HypreDriverStruct::printMatrix() begin" << "\n";
  if (!_params->printSystem) return;
  HYPRE_StructMatrixPrint((fileName + ".sstruct").c_str(), _HA, 0);
  //  if (_requiresPar) {
  //    HYPRE_ParCSRMatrixPrint(_HA_Par, (fileName + ".par").c_str());
  //    // Print CSR matrix in IJ format, base 1 for rows and cols
  //    HYPRE_ParCSRMatrixPrintIJ(_HA_Par, 1, 1, (fileName + ".ij").c_str());
  //  }
  cout_doing << "HypreDriverStruct::printMatrix() end" << "\n";
}

void
HypreDriverStruct::printRHS(const string& fileName /* =  "output_b" */)
{
  if (!_params->printSystem) return;
  HYPRE_StructVectorPrint(fileName.c_str(), _HB, 0);
  //  if (_requiresPar) {
  //    HYPRE_ParVectorPrint(_HB_Par, (fileName + ".par").c_str());
  //  }
}

void
HypreDriverStruct::printSolution(const string& fileName /* =  "output_x" */)
{
  if (!_params->printSystem) return;
  HYPRE_StructVectorPrint(fileName.c_str(), _HX, 0);
  //  if (_requiresPar) {
  //    HYPRE_ParVectorPrint(_HX_Par, (fileName + ".par").c_str());
  //  }
}

void
HypreDriverStruct::gatherSolutionVector(void)
{
  // It seems it is not necessary to gather the solution vector
  // for the Struct interface.
} // end HypreDriverStruct::gatherSolutionVector()

//#####################################################################
// class HypreDriver implementation for CC variable type
//#####################################################################

void
HypreDriverStruct::makeLinearSystem_CC(const int matl)
  //___________________________________________________________________
  // Function HypreDriverStruct::makeLinearSystem_CC~
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
  HYPRE_StructGridCreate(_pg->getComm(), 3, &_grid);

  for(int p=0;p<_patches->size();p++){
    const Patch* patch = _patches->get(p);
    Patch::VariableBasis basis =
      Patch::translateTypeToBasis(sol_type::getTypeDescription()
                                  ->getType(), true);
    IntVector ec = _params->getSolveOnExtraCells() ?
      IntVector(0,0,0) : -_level->getExtraCells();
    IntVector l = patch->getLowIndex(basis, ec);
    IntVector h1 = patch->getHighIndex(basis, ec)-IntVector(1,1,1);

    HYPRE_StructGridSetExtents(_grid, l.get_pointer(), h1.get_pointer());
  }
  HYPRE_StructGridAssemble(_grid);

  //==================================================================
  // Set up the stencil
  //==================================================================
  if(_params->symmetric){
    HYPRE_StructStencilCreate(3, 4, &_stencil);
    int offsets[4][3] = {{0,0,0},
                         {-1,0,0},
                         {0,-1,0},
                         {0,0,-1}};
    for(int i=0;i<4;i++)
      HYPRE_StructStencilSetElement(_stencil, i, offsets[i]);
  } else {
    HYPRE_StructStencilCreate(3, 7, &_stencil);
    int offsets[7][3] = {{0,0,0},
                         {1,0,0}, {-1,0,0},
                         {0,1,0}, {0,-1,0},
                         {0,0,1}, {0,0,-1}};
    for(int i=0;i<7;i++)
      HYPRE_StructStencilSetElement(_stencil, i, offsets[i]);
  }

  //==================================================================
  // Set up the Struct left-hand-side matrix _HA
  //==================================================================
  HYPRE_StructMatrixCreate(_pg->getComm(), _grid, _stencil, &_HA);
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

} // end HypreDriverStruct::makeLinearSystem_CC()


void
HypreDriverStruct::getSolution_CC(const int matl)
  //_____________________________________________________________________
  // Function HypreDriverStruct::getSolution_CC~
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
} // end HypreDriverStruct::getSolution_CC()
