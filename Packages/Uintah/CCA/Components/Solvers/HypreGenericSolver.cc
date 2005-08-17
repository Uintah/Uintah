#include <Packages/Uintah/CCA/Components/Solvers/HypreGenericSolver.h>
#include <Packages/Uintah/CCA/Components/Solvers/HypreSolverPFMG.h>
#include <Packages/Uintah/CCA/Components/Solvers/HypreDriverStruct.h>
#include <Packages/Uintah/CCA/Components/Solvers/HypreDriverSStruct.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Core/Util/DebugStream.h>

using namespace Uintah;
//__________________________________
//  To turn on normal output
//  setenv SCI_DEBUG "HYPRE_DOING_COUT:+"

static DebugStream cout_doing("HYPRE_DOING_COUT", false);

HypreGenericSolver::HypreGenericSolver(HypreDriver* driver,
                                       HypreGenericPrecond* precond,
                                       const Priorities& priority) :
  _driver(driver), _precond(precond), _priority(priority), _requiresPar(true)
{
  assertInterface();

  // Initialize results section
  _results.numIterations = 0;
  _results.finalResNorm  = 1.23456e+30; // Large number
}
  
HypreGenericSolver::~HypreGenericSolver(void)
{
}
  
void
HypreGenericSolver::assertInterface(void)
{ 
  if (_priority.size() < 1) {
    throw InternalError("Solver created without interface priorities",
                        __FILE__, __LINE__);
 
  }

  // Intersect solver and preconditioner priorities
  if (_precond) {
    Priorities newSolverPriority;
    const Priorities& precondPriority = _precond->getPriority();
    for (unsigned int i = 0; i < _priority.size(); i++) {
      bool remove = false;
      for (unsigned int j = 0; j < priority.size(); j++) {
        if (_priority[i] == precondPriority[j]) {
          // Remove this solver interface entry
          remove = true;
          break;
        }
        if (!remove) {
          newSolverPriority.push_back(_priority[i]);
        }
      }
    }
    _priority = newSolverPriority;
  } // end if (_precond)

  // Check whether solver requires ParCSR or not, because we need to
  // know about that in HypreDriver::makeLinearSystem. Also check the
  // correctness of the values of _priority.
  for (unsigned int i = 0; i < _priority.size(); i++) {
    if (_priority[i] == HypreInterfaceNA) {
      throw InternalError("Bad Solver interface priority "+_priority[i],
                          __FILE__, __LINE__);
    } else if ((_priority[i] != HypreParCSR) && (_requiresPar)) {
      // Modify this rule if we use other Hypre interfaces in the future.
      // See HypreTypes.h.
      _requiresPar = false;
    }
  }

  const HypreInterface& interface = _driver->getInterface();
  bool found = false;
  for (unsigned int i = 0; i < _priority.size(); i++) {
    if (interface == _priority[i]) {
      // Found interface that solver can work with
      found = true;
      break;
    }
  }

  // See whether we can convert the Hypre data to a format we can
  // work with.
  if (!found) {
    for (unsigned int i = 0; i < _priority.size(); i++) {
      // Try to convert from the current driver to _priority[i]
      if (_driver->isConvertable(_priority[i])) {
        // Conversion exists
        found = true;
        break;
      }
    }
  }

  if (!found) {
    throw InternalError("Solver does not support Hypre interface "+interface,
                        __FILE__, __LINE__); 
  }
}

HypreGenericSolver*
newHypreSolver(const SolverType& solverType,
               const PrecondType& precondType,
               HypreDriver* driver)
  // Create a new solver object of specific solverType solver type
  // but a generic solver pointer type.
  // TODO: include all derived classes here.
{
  const Priorities precondPriority;
  switch (solverType) {
#if 0
  case SMG:
    {
      return new HypreSolverSMG(driver);
    }
#endif
  case PFMG:
    {
      return new HypreSolverPFMG(driver,precondPriority);
    }
#if 0
  case SparseMSG:
    {
      return new HypreSolverSparseMSG(driver);
    }
  case CG:
    {
      return new HypreSolverCG(driver);
    }
  case Hybrid: 
    {
      return new HypreSolverHybrid(driver);
    }
  case GMRES:
    {
      return new HypreSolverGMRES(driver);
    }
#endif
  default:
    throw InternalError("Unsupported solver type: "+solverType,
                        __FILE__, __LINE__);
  } // switch (solverType)
  return 0;
}

SolverType
getSolverType(const string& solverTitle)
{
  // Determine solver type from title
  if ((solverTitle == "SMG") ||
      (solverTitle == "smg")) {
    return SMG;
  } else if ((solverTitle == "PFMG") ||
             (solverTitle == "pfmg")) {
    return PFMG;
  } else if ((solverTitle == "SparseMSG") ||
             (solverTitle == "sparsemsg")) {
    return SparseMSG;
  } else if ((solverTitle == "CG") ||
             (solverTitle == "cg") ||
             (solverTitle == "PCG") ||
             (solverTitle == "conjugategradient")) {
    return CG;
  } else if ((solverTitle == "Hybrid") ||
             (solverTitle == "hybrid")) {
    return Hybrid;
  } else if ((solverTitle == "GMRES") ||
             (solverTitle == "gmres")) {
    return GMRES;
  } else if ((solverTitle == "AMG") ||
             (solverTitle == "amg") ||
             (solverTitle == "BoomerAMG") ||
             (solverTitle == "boomeramg")) {
    return AMG;
  } else if ((solverTitle == "FAC") ||
             (solverTitle == "fac")) {
    return FAC;
  } else {
    throw InternalError("Unknown solver type: "+solverTitle,
                        __FILE__, __LINE__);
  } // end "switch" (solverTitle)
} // end solverFromTitle()
