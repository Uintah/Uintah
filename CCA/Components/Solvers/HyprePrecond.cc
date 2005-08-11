//--------------------------------------------------------------------------
// File: HyprePrecond.cc
// 
// A generic Hypre preconditioner driver that checks whether the precond
// can work with the input interface. The actual precond setup/destroy is
// done in the classes derived from HyprePrecond.
//--------------------------------------------------------------------------
#include <Packages/Uintah/CCA/Components/Solvers/HyprePrecond.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Core/Util/DebugStream.h>

using namespace Uintah;
//__________________________________
//  To turn on normal output
//  setenv SCI_DEBUG "HYPRE_DOING_COUT:+"

static DebugStream cout_doing("HYPRE_DOING_COUT", false);

namespace Uintah {

  HyprePrecond::HyprePrecond(const HypreInterface& interface,
                             const int acceptableInterface)
    : _interface(interface)
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
    for (HypreInterface interface = HypreStruct;
         interface <= HypreParCSR; ++interface) {
      if ((acceptableInterface & interface) && (_interface == interface)) {
        return;
      }
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
    default:
      throw InternalError("Unknown preconditionertype in newHyprePrecond: "
                          +precondType, __FILE__, __LINE__);

    } // switch (precondType)
    return 0;
  } // end newHyprePrecond()

} // end namespace Uintah
