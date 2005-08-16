//--------------------------------------------------------------------------
// File: HyprePrecond.cc
// 
// A generic Hypre preconditioner driver that checks whether the precond
// can work with the input interface. The actual precond setup/destroy is
// done in the classes derived from HyprePrecond.
//--------------------------------------------------------------------------
#include <Packages/Uintah/CCA/Components/Solvers/HyprePrecond.h>
#include <Packages/Uintah/CCA/Components/Solvers/HyprePrecondSMG.h>
#include <Packages/Uintah/CCA/Components/Solvers/HypreSolverParams.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
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

namespace Uintah {

  HyprePrecond::HyprePrecond(const HypreInterface& interface,
                             const ProcessorGroup* pg,
                             const HypreSolverParams* params,
                             const int acceptableInterface) :
    _interface(interface), _pg(pg), _params(params)
  { 
    assertInterface(acceptableInterface);
    this->setup(); // Derived class setup()
  }

  HyprePrecond::~HyprePrecond(void)
  { 
    this->destroy(); // Derived class destroy()
  }
  
  void
  HyprePrecond::assertInterface(const int acceptableInterface)
  { 
    if (acceptableInterface & _interface) {
      return;
    }
    throw InternalError("Preconditioner does not support Hypre interface: "
                        +_interface,__FILE__, __LINE__);
  }

  PrecondType   
  precondFromTitle(const std::string& precondTitle)
  {
    /* Determine preconditioner type from title */
    if ((precondTitle == "SMG") ||
        (precondTitle == "smg")) {
      return PrecondSMG;
    } else if ((precondTitle == "PFMG") ||
               (precondTitle == "pfmg")) {
      return PrecondPFMG;
    } else if ((precondTitle == "SparseMSG") ||
               (precondTitle == "sparsemsg")) {
      return PrecondSparseMSG;
    } else if ((precondTitle == "Jacobi") ||
               (precondTitle == "jacobi")) {
      return PrecondJacobi;
    } else if ((precondTitle == "Diagonal") ||
               (precondTitle == "diagonal")) {
      return PrecondDiagonal;
    } else if ((precondTitle == "AMG") ||
               (precondTitle == "amg") ||
               (precondTitle == "BoomerAMG") ||
               (precondTitle == "boomeramg")) {
      return PrecondAMG;
    } else if ((precondTitle == "FAC") ||
               (precondTitle == "fac")) {
      return PrecondFAC;
    } else {
      throw InternalError("Unknown preconditionertype: "+precondTitle,
                          __FILE__, __LINE__);
    } // end "switch" (precondTitle)
  } // end precondFromTitle()

  HyprePrecond*
  newHyprePrecond(const PrecondType& precondType,
                  const HypreInterface& interface,
                  const ProcessorGroup* pg,
                  const HypreSolverParams* params)
    // Create a new preconditioner object of specific precond type
    // "precondType" but a generic preconditioner pointer type.
  {
    switch (precondType) {
    case PrecondNA:
      {
        // No preconditioner
        return 0;
      }
    case PrecondSMG:
      {
        return new HyprePrecondSMG(interface,pg,params);
      }
#if 0
    case PrecondPFMG:
      {
        return new HyprePrecondPFMG();
      }
    case PrecondSparseMSG:
      {
        return new HyprePrecondSparseMSG();
      }
    case PrecondJacobi:
      {
        return new HyprePrecondJacobi();
      }
    case PrecondDiagonal:
      {
        return new HyprePrecondDiagonal();
      }
#endif
    default:
      throw InternalError("Unknown preconditionertype in newHyprePrecond: "
                          +precondType, __FILE__, __LINE__);

    } // switch (precondType)
    return 0;
  } // end newHyprePrecond()


  PrecondType
  getPrecondType(const string& precondTitle)
  {
    /* Determine preconditioner type from title */
    if ((precondTitle == "SMG") ||
        (precondTitle == "smg")) {
      return PrecondSMG;
    } else if ((precondTitle == "PFMG") ||
               (precondTitle == "pfmg")) {
      return PrecondPFMG;
    } else if ((precondTitle == "SparseMSG") ||
               (precondTitle == "sparsemsg")) {
      return PrecondSparseMSG;
    } else if ((precondTitle == "Jacobi") ||
               (precondTitle == "jacobi")) {
      return PrecondJacobi;
    } else if ((precondTitle == "Diagonal") ||
               (precondTitle == "diagonal")) {
      return PrecondDiagonal;
    } else if ((precondTitle == "AMG") ||
               (precondTitle == "amg") ||
               (precondTitle == "BoomerAMG") ||
               (precondTitle == "boomeramg")) {
      return PrecondAMG;
    } else if ((precondTitle == "FAC") ||
               (precondTitle == "fac")) {
      return PrecondFAC;
    } else {
      throw InternalError("Unknown preconditionertype: "+precondTitle,
                          __FILE__, __LINE__);
    } // end "switch" (precondTitle)
  } // end precondFromTitle()

#if 0
  template<class Types>
  void HypreDriver<Types>::setupPrecond(const ProcessorGroup* pg,
                                        HYPRE_PtrToSolverFcn& precond,
                                        HYPRE_PtrToSolverFcn& pcsetup,
                                        HYPRE_StructSolver& precond_solver)
    /*_____________________________________________________________________
      Function HypreDriver::setupPrecond
      Set up and initialize the Hypre preconditioner, if we use one.
      _____________________________________________________________________*/
  {
    switch (params->precondType) {
    case HypreSolverParams::PrecondNA:
      {
        /* No preconditioner, do nothing */
        break;
      } // case HypreSolverParams::PrecondNA

    case HypreSolverParams::PrecondSMG:
      /* use symmetric SMG as preconditioner */
      {
        HYPRE_StructSMGCreate(pg->getComm(), &precond_solver);
        HYPRE_StructSMGSetMemoryUse(precond_solver, 0);
        HYPRE_StructSMGSetMaxIter(precond_solver, 1);
        HYPRE_StructSMGSetTol(precond_solver, 0.0);
        HYPRE_StructSMGSetZeroGuess(precond_solver);
        HYPRE_StructSMGSetNumPreRelax(precond_solver, params->nPre);
        HYPRE_StructSMGSetNumPostRelax(precond_solver, params->nPost);
        HYPRE_StructSMGSetLogging(precond_solver, 0);
        precond = (HYPRE_PtrToSolverFcn)HYPRE_StructSMGSolve;
        pcsetup = (HYPRE_PtrToSolverFcn)HYPRE_StructSMGSetup;
        break;
      } // case HypreSolverParams::PrecondSMG

    case HypreSolverParams::PrecondPFMG:
      /* use symmetric PFMG as preconditioner */
      {
        HYPRE_StructPFMGCreate(pg->getComm(), &precond_solver);
        HYPRE_StructPFMGSetMaxIter(precond_solver, 1);
        HYPRE_StructPFMGSetTol(precond_solver, 0.0);
        HYPRE_StructPFMGSetZeroGuess(precond_solver);
        /* weighted Jacobi = 1; red-black GS = 2 */
        HYPRE_StructPFMGSetRelaxType(precond_solver, 1);
        HYPRE_StructPFMGSetNumPreRelax(precond_solver, params->nPre);
        HYPRE_StructPFMGSetNumPostRelax(precond_solver, params->nPost);
        HYPRE_StructPFMGSetSkipRelax(precond_solver, params->skip);
        HYPRE_StructPFMGSetLogging(precond_solver, 0);
        precond = (HYPRE_PtrToSolverFcn)HYPRE_StructPFMGSolve;
        pcsetup = (HYPRE_PtrToSolverFcn)HYPRE_StructPFMGSetup;
        break;
      } // case HypreSolverParams::PrecondPFMG

    case HypreSolverParams::PrecondSparseMSG:
      /* use symmetric SparseMSG as preconditioner */
      {
        HYPRE_StructSparseMSGCreate(pg->getComm(), &precond_solver);
        HYPRE_StructSparseMSGSetMaxIter(precond_solver, 1);
        HYPRE_StructSparseMSGSetJump(precond_solver, params->jump);
        HYPRE_StructSparseMSGSetTol(precond_solver, 0.0);
        HYPRE_StructSparseMSGSetZeroGuess(precond_solver);
        /* weighted Jacobi = 1; red-black GS = 2 */
        HYPRE_StructSparseMSGSetRelaxType(precond_solver, 1);
        HYPRE_StructSparseMSGSetNumPreRelax(precond_solver, params->nPre);
        HYPRE_StructSparseMSGSetNumPostRelax(precond_solver, params->nPost);
        HYPRE_StructSparseMSGSetLogging(precond_solver, 0);
        precond = (HYPRE_PtrToSolverFcn)HYPRE_StructSparseMSGSolve;
        pcsetup = (HYPRE_PtrToSolverFcn)HYPRE_StructSparseMSGSetup;
        break;
      } // case HypreSolverParams::PrecondSparseMSG

    case HypreSolverParams::PrecondJacobi:
      /* use two-step Jacobi as preconditioner */
      {
        HYPRE_StructJacobiCreate(pg->getComm(), &precond_solver);
        HYPRE_StructJacobiSetMaxIter(precond_solver, 2);
        HYPRE_StructJacobiSetTol(precond_solver, 0.0);
        HYPRE_StructJacobiSetZeroGuess(precond_solver);
        precond = (HYPRE_PtrToSolverFcn)HYPRE_StructJacobiSolve;
        pcsetup = (HYPRE_PtrToSolverFcn)HYPRE_StructJacobiSetup;
        break;
      } // case HypreSolverParams::PrecondJacobi

    case HypreSolverParams::PrecondDiagonal:
      /* use diagonal scaling as preconditioner */
      {
#ifdef HYPRE_USE_PTHREADS
        for (i = 0; i < hypre_NumThreads; i++)
          precond[i] = NULL;
#else
        precond = NULL;
#endif
        precond = (HYPRE_PtrToSolverFcn)HYPRE_StructDiagScale;
        pcsetup = (HYPRE_PtrToSolverFcn)HYPRE_StructDiagScaleSetup;
        break;
      } // case HypreSolverParams::PrecondDiagonal

    default:
      // This should have been caught in readParameters...
      throw InternalError("Unknown preconditionertype: "
                          +params->precondTitle,
                          __FILE__, __LINE__);
    } // end switch (param->precondType)
  } // end setupPrecond()

  template<class Types>
  void HypreDriver<Types>::destroyPrecond
  (HYPRE_StructSolver& precond_solver)
    /*_____________________________________________________________________
      Function HypreDriver::destroyPrecond
      Destroy (+free) the Hypre preconditioner.
      _____________________________________________________________________*/
  {
    switch (params->precondType) {
    case HypreSolverParams::PrecondNA:
      {
        /* No preconditioner, do nothing */
        break;
      } // case HypreSolverParams::PrecondNA
    case HypreSolverParams::PrecondSMG:
      {
        HYPRE_StructSMGDestroy(precond_solver);
        break;
      } // case HypreSolverParams::PrecondSMG

    case HypreSolverParams::PrecondPFMG:
      {
        HYPRE_StructPFMGDestroy(precond_solver);
        break;
      } // case HypreSolverParams::PrecondPFMG

    case HypreSolverParams::PrecondSparseMSG:
      {
        HYPRE_StructSparseMSGDestroy(precond_solver);
        break;
      } // case HypreSolverParams::PrecondSparseMSG
      
    case HypreSolverParams::PrecondJacobi:
      {
        HYPRE_StructJacobiDestroy(precond_solver);
        break;
      } // case HypreSolverParams::PrecondJacobi

    case HypreSolverParams::PrecondDiagonal:
      /* Nothing to destroy for diagonal preconditioner */
      {
        break;
      } // case HypreSolverParams::PrecondDiagonal

    default:
      // This should have been caught in readParameters...
      throw InternalError("Unknown preconditionertype in destroyPrecond: "
                          +params->precondType, __FILE__, __LINE__);
    } // end switch (param->precondType)
  } // end destroyPrecond()

#endif

} // end namespace Uintah
