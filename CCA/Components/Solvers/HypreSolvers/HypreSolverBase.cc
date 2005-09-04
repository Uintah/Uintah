#include <Packages/Uintah/CCA/Components/Solvers/HypreSolvers/HypreSolverBase.h>
#include <Packages/Uintah/CCA/Components/Solvers/HypreSolvers/HypreSolverSMG.h>
#include <Packages/Uintah/CCA/Components/Solvers/HypreSolvers/HypreSolverPFMG.h>
#include <Packages/Uintah/CCA/Components/Solvers/HypreSolvers/HypreSolverSparseMSG.h>
#include <Packages/Uintah/CCA/Components/Solvers/HypreSolvers/HypreSolverCG.h>
#include <Packages/Uintah/CCA/Components/Solvers/HypreSolvers/HypreSolverHybrid.h>
#include <Packages/Uintah/CCA/Components/Solvers/HypreSolvers/HypreSolverGMRES.h>
#include <Packages/Uintah/CCA/Components/Solvers/HypreSolvers/HypreSolverAMG.h>
#include <Packages/Uintah/CCA/Components/Solvers/HypreSolvers/HypreSolverFAC.h>
#include <Packages/Uintah/CCA/Components/Solvers/HypreDriverStruct.h>
#include <Packages/Uintah/CCA/Components/Solvers/HypreDriverSStruct.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Core/Util/DebugStream.h>

#include <sci_comp_warn_fixes.h>

using namespace Uintah;
//__________________________________
//  To turn on normal output
//  setenv SCI_DEBUG "HYPRE_DOING_COUT:+"

static DebugStream cout_doing("HYPRE_DOING_COUT", false);

HypreSolverBase::HypreSolverBase(HypreDriver* driver,
                                 HyprePrecondBase* precond,
                                 const Priorities& priority) :
  _driver(driver), _precond(precond), _priority(priority), _requiresPar(true)
{
  cerr << "HypreSolverBase::constructor BEGIN" << "\n";
  assertInterface();

  // Initialize results section
  _results.numIterations = 0;
  _results.finalResNorm  = 1.23456e+30; // Large number
  cerr << "HypreSolverBase::constructor END" << "\n";
}
  
HypreSolverBase::~HypreSolverBase(void)
{
}
  
void
HypreSolverBase::assertInterface(void)
{ 
  cerr << "HypreSolverBase::assertInterface() BEGIN" << "\n";
  if (_priority.size() < 1) {
    throw InternalError("Solver created without interface priorities",
                        __FILE__, __LINE__);
 
  }

  // Intersect solver and preconditioner priorities
  if (_precond) {
    cerr << "Intersect solver, precond priorities begin" << "\n";
    Priorities newSolverPriority;
    const Priorities& precondPriority = _precond->getPriority();
    for (unsigned int i = 0; i < _priority.size(); i++) {
      cerr << "i = " << i << "\n";
      bool remove = false;
      for (unsigned int j = 0; j < _priority.size(); j++) {
        cerr << "j = " << j << "\n";
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
  cerr << "Check if solver requires par" << "\n";
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
  cerr << "requiresPar = " << _requiresPar << "\n";

  cerr << "Look for requested interface in solver priorities" << "\n";
  const HypreInterface& interface = _driver->getInterface();
  cerr << "interface = " << interface << "\n";
  bool found = false;
  cerr << "Solver priorities:" << "\n";
  for (unsigned int i = 0; i < _priority.size(); i++) {
    cerr << "_priority[" << i << "] = " << _priority[i] << "\n";
    if (interface == _priority[i]) {
      // Found interface that solver can work with
      found = true;
      break;
    }
  }
  cerr << "1. found = " << found << "\n";

  // See whether we can convert the Hypre data to a format we can
  // work with.
  if (!found) {
    cerr << "Looking for possible conversions" << "\n";
    for (unsigned int i = 0; i < _priority.size(); i++) {
      cerr << "i = " << i << "\n";
      // Try to convert from the current driver to _priority[i]
      if (_driver->isConvertable(_priority[i])) {
        // Conversion exists
        found = true;
        break;
      }
    }
  }
  cerr << "2. found = " << found << "\n";

  if (!found) {
    ostringstream msg;
    msg << "Solver does not support Hypre interface " << interface;
    throw InternalError(msg.str(),__FILE__, __LINE__); 
  }
  cerr << "HypreSolverBase::assertInterface() END" << "\n";
}

namespace Uintah {

  HypreSolverBase*
  newHypreSolver(const SolverType& solverType,
                 HypreDriver* driver,
                 HyprePrecondBase* precond)
    // Create a new solver object of specific solverType solver type
    // but a generic solver pointer type.
    // Include all derived solver classes here.
  {
    cerr << "newHypreSolver() BEGIN" << "\n";
    const Priorities precondPriority;
    switch (solverType) {
    case SMG:
      {
        return new HypreSolverSMG(driver,precond);
      }
    case PFMG:
      {
        return new HypreSolverPFMG(driver,precond);
      }
    case SparseMSG:
      {
        return new HypreSolverSparseMSG(driver,precond);
      }
    case CG:
      {
        cerr << "Doing new HypreSolverCG" << "\n";
        return new HypreSolverCG(driver,precond);
      }
    case Hybrid: 
      {
        return new HypreSolverHybrid(driver,precond);
      }
    case GMRES:
      {
        return new HypreSolverGMRES(driver,precond);
      }
    case AMG:
      {
        return new HypreSolverAMG(driver,precond);
      }
    case FAC:
      {
        return new HypreSolverFAC(driver,precond);
      }
    default:
      throw InternalError("Unsupported solver type: "+solverType,
                          __FILE__, __LINE__);
    } // switch (solverType)
    cerr << "newHypreSolver() END (shouldn't be reached)" << "\n";
    RETURN_0;
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

  ostream&
  operator << (ostream& os, const SolverType& solverType)
    // Write a solver type (enum) to the stream os.
  {
    switch (solverType) {
    case SMG:       { os << "SMG"; break; }
    case PFMG:      { os << "PFMG"; break; }
    case SparseMSG: { os << "SparseMSG"; break; }
    case CG:        { os << "CG";  break; }
    case Hybrid:    { os << "Hybrid"; break; }
    case GMRES:     { os << "GMRES"; break; }
    case AMG:       { os << "AMG"; break; }
    case FAC:       { os << "FAC"; break; }
    default:        { os << "???"; break; }
    } // switch (solverType)

    return os;
  }

} // end namespace Uintah
