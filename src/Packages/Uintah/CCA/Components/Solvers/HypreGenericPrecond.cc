//--------------------------------------------------------------------------
// File: HypreGenericPrecond.cc
// 
// A generic Hypre preconditioner driver that checks whether the precond
// can work with the input interface. The actual precond setup/destroy is
// done in the classes derived from HypreGenericPrecond.
//--------------------------------------------------------------------------
#include <Packages/Uintah/CCA/Components/Solvers/HypreGenericPrecond.h>
#include <Packages/Uintah/CCA/Components/Solvers/HyprePrecondSMG.h>
#include <Packages/Uintah/CCA/Components/Solvers/HyprePrecondPFMG.h>
#include <Packages/Uintah/CCA/Components/Solvers/HyprePrecondSparseMSG.h>
#include <Packages/Uintah/CCA/Components/Solvers/HyprePrecondJacobi.h>
#include <Packages/Uintah/CCA/Components/Solvers/HyprePrecondDiagonal.h>
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

HypreGenericPrecond::HypreGenericPrecond(const Priorities& priority) :
  _priority(priority)
{
  if (_priority.size() < 1) {
    throw InternalError("Preconditioner created without interface "
                        "priorities",
                        __FILE__, __LINE__);
  }
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

HypreGenericPrecond*
newHyprePrecond(const PrecondType& precondType)
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
      return new HyprePrecondSMG();
    }
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
  case PrecondAMG:
    {
      break; // Not implemented yet
    }
  case PrecondFAC:
    {
      break; // Not implemented yet
    }
  default:
    throw InternalError("Unknown preconditionertype in newHyprePrecond: "
                        +precondType, __FILE__, __LINE__);

  } // switch (precondType)
  throw InternalError("Preconditioner not yet implemented in newHyprePrecond: "
                      +precondType, __FILE__, __LINE__);
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
