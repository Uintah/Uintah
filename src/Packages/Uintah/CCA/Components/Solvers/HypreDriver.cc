/*--------------------------------------------------------------------------
 * File: HypreDriver.cc
 *
 * Implementation of a wrapper of a Hypre solver for a particular variable
 * type. 
 *--------------------------------------------------------------------------*/
// TODO: 
// * Separate setup from solver phase in solvers. Some solvers can be
//   more efficient by separating the setup phase that depends
//   on A only, from the solution stage, that depends on A, b, x.

#include <sci_defs/hypre_defs.h>

#if HAVE_HYPRE_1_9
#include <Packages/Uintah/CCA/Components/Solvers/HypreDriver.h>
#include <Packages/Uintah/CCA/Components/Solvers/HypreDriverStruct.h>
#include <Packages/Uintah/CCA/Components/Solvers/HypreDriverSStruct.h>
#include <Packages/Uintah/CCA/Components/Solvers/MatrixUtil.h>
#include <Packages/Uintah/Core/Grid/Level.h>
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
  class HypreDriver implementation common to all variable types
  and all Hypre interfaces
  _____________________________________________________________________*/

void
HypreDriver::makeLinearSystem_SFCX(const int matl)
{
  throw InternalError("makeLinearSystem is not implemented for SFCX variables",
                      __FILE__, __LINE__);
}

void
HypreDriver::makeLinearSystem_SFCY(const int matl)
{
  throw InternalError("makeLinearSystem is not implemented for SFCY variables",
                      __FILE__, __LINE__);
}

void
HypreDriver::makeLinearSystem_SFCZ(const int matl)
{
  throw InternalError("makeLinearSystem is not implemented for SFCZ variables",
                      __FILE__, __LINE__);
}

void
HypreDriver::makeLinearSystem_NC(const int matl)
{
  throw InternalError("makeLinearSystem is not implemented for NC variables",
                      __FILE__, __LINE__);
}
void
HypreDriver::makeLinearSystem_CC(const int matl)
{
  throw InternalError("makeLinearSystem is not implemented for CC variables",
                      __FILE__, __LINE__);
}

void
HypreDriver::getSolution_SFCX(const int matl)
{
  throw InternalError("getSolution is not implemented for SFCX variables",
                      __FILE__, __LINE__);
}

void
HypreDriver::getSolution_SFCY(const int matl)
{
  throw InternalError("getSolution is not implemented for SFCY variables",
                      __FILE__, __LINE__);
}

void
HypreDriver::getSolution_SFCZ(const int matl)
{
  throw InternalError("getSolution is not implemented for SFCZ variables",
                      __FILE__, __LINE__);
}

void
HypreDriver::getSolution_NC(const int matl)
{
  throw InternalError("getSolution is not implemented for NC variables",
                      __FILE__, __LINE__);
}
void
HypreDriver::getSolution_CC(const int matl)
{
  throw InternalError("getSolution is not implemented for CC variables",
                      __FILE__, __LINE__);
}

bool
HypreDriver::isConvertable(const HypreInterface& to)
  // Is it possible to convert from the current HypreInterface
  // "_interface" to "to" ?
{
  // Add here any known rules of conversion.

  // It does not seem possible to convert from Struct to ParCSR.
  //  if ((_interface == HypreStruct ) && (to == HypreParCSR)) return true;

  if ((_interface == HypreSStruct) && (to == HypreParCSR)) return true;

  return false;
} // end isConvertable()

//_____________________________________________________________________
// Function TypeTemplate2Enum~
// Convert a template variable type to an enum variarable type.
// Add instantiations for every new variable type that is later added.
//_____________________________________________________________________

namespace Uintah {

  template<>
  TypeDescription::Type
  TypeTemplate2Enum(const SFCXTypes& /*t*/)
  {
    return TypeDescription::SFCXVariable;
  }

  template<>
  TypeDescription::Type
  TypeTemplate2Enum(const SFCYTypes& /*t*/)
  {
    return TypeDescription::SFCYVariable;
  }

  template<>
  TypeDescription::Type
  TypeTemplate2Enum(const SFCZTypes& /*t*/)
  {
    return TypeDescription::SFCZVariable;
  }

  template<>
  TypeDescription::Type
  TypeTemplate2Enum(const NCTypes& /*t*/)
  {
    return TypeDescription::NCVariable;
  }

  template<>
  TypeDescription::Type
  TypeTemplate2Enum(const CCTypes& /*t*/)
  {
    return TypeDescription::CCVariable;
  }

  HypreDriver*
  newHypreDriver(const HypreInterface& interface,
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
    case HypreStruct: 
      {
        return new HypreDriverStruct
          (level, matlset, A, which_A_dw,
           x, modifies_x, b, which_b_dw, guess, 
           which_guess_dw, params, interface);
      }
    case HypreSStruct:
      {
        return new HypreDriverSStruct
          (level, matlset, A, which_A_dw,
           x, modifies_x, b, which_b_dw, guess, 
           which_guess_dw, params, interface);
      }
    default:
      throw InternalError("Unsupported Hypre Interface: "+interface,
                          __FILE__, __LINE__);
    } // end switch (interface)
  }

  //_____________________________________________________________________
  // HypreInterface operations
  //_____________________________________________________________________

  HypreInterface& operator++(HypreInterface &s)
  {
    return s = HypreInterface(2*s);
  }
  
  std::ostream&
  operator << (std::ostream& os, const HypreInterface& i)
    // Write Hypre interface i to the stream os.
  {
    if      (i == HypreStruct     ) os << "HypreStruct (1)";
    else if (i == HypreSStruct    ) os << "HypreSStruct (2)";
    else if (i == HypreParCSR     ) os << "HypreParCSR (4)";
    else if (i == HypreInterfaceNA) os << "HypreInterfaceNA (8)";
    else os << "N/A";
    return os;
  }

  //_____________________________________________________________________
  // BoxSide operations
  //_____________________________________________________________________

  BoxSide& operator++(BoxSide &s)
  {
    return s = BoxSide(s+2);
  }
  
  std::ostream&
  operator << (std::ostream& os, const BoxSide& s)
    // Write side s to the stream os.
  {
    if      (s == LeftSide ) os << "Left ";
    else if (s == RightSide) os << "Right";
    else os << "N/A";
    return os;
  }

  BoxSide
  patchFaceSide(const Patch::FaceType& patchFace)
    // Correspondence between Uintah patch face and their BoxSide
    // (left or right).
  {
    if (patchFace == Patch::xminus || 
        patchFace == Patch::yminus || 
        patchFace == Patch::zminus) {
      return LeftSide;
    } else if (patchFace == Patch::xplus || 
               patchFace == Patch::yplus || 
               patchFace == Patch::zplus){
      return RightSide;
    } else {
      ostringstream msg;
      msg << "patchFaceSide() called with invalid Patch::FaceType "
          << patchFace;
      throw InternalError(msg.str(),__FILE__, __LINE__);
    }
  }

} // end namespace Uintah

#endif // HAVE_HYPRE_1_9
