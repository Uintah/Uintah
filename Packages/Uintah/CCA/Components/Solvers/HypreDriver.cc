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
  throw InternalError("solve() is not implemented for SFCX variables",
                      __FILE__, __LINE__);
}

void
HypreDriver::makeLinearSystem_SFCY(const int matl)
{
  throw InternalError("solve() is not implemented for SFCY variables",
                      __FILE__, __LINE__);
}

void
HypreDriver::makeLinearSystem_SFCZ(const int matl)
{
  throw InternalError("solve() is not implemented for SFCZ variables",
                      __FILE__, __LINE__);
}

void
HypreDriver::makeLinearSystem_NC(const int matl)
{
  throw InternalError("solve() is not implemented for NC variables",
                      __FILE__, __LINE__);
}
void
HypreDriver::makeLinearSystem_CC(const int matl)
{
  throw InternalError("solve() is not implemented for CC variables",
                      __FILE__, __LINE__);
}

void
HypreDriver::getSolution_SFCX(const int matl)
{
  throw InternalError("solve() is not implemented for SFCX variables",
                      __FILE__, __LINE__);
}

void
HypreDriver::getSolution_SFCY(const int matl)
{
  throw InternalError("solve() is not implemented for SFCY variables",
                      __FILE__, __LINE__);
}

void
HypreDriver::getSolution_SFCZ(const int matl)
{
  throw InternalError("solve() is not implemented for SFCZ variables",
                      __FILE__, __LINE__);
}

void
HypreDriver::getSolution_NC(const int matl)
{
  throw InternalError("solve() is not implemented for NC variables",
                      __FILE__, __LINE__);
}
void
HypreDriver::getSolution_CC(const int matl)
{
  throw InternalError("solve() is not implemented for CC variables",
                      __FILE__, __LINE__);
}

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

} // end namespace Uintah

HypreInterface& operator++(HypreInterface &s)
{
  return s = HypreInterface(2*s);
}
  
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

void
printValues(const int stencilSize,
            const int numCells,
            const double* values /* = 0 */,
            const double* rhsValues /* = 0 */,
            const double* solutionValues /* = 0 */)
  /* Print values, rhsValues vectors */
{
  cout_doing << "--- Printing values,rhsValues,solutionValues arrays ---" << "\n";
  for (int cell = 0; cell < numCells; cell++) {
    int offsetValues    = stencilSize * cell;
    int offsetRhsValues = cell;
    cout_doing << "cell = " << cell << "\n";
    if (values) {
      for (int entry = 0; entry < stencilSize; entry++) {
        cout_doing << "values   [" << offsetValues + entry
                   << "] = "  << values[offsetValues + entry]
                   << "\n";
      }
    }
    if (rhsValues) {
      cout_doing << "rhsValues[" << offsetRhsValues 
                 << "] = " << rhsValues[offsetRhsValues] << "\n";
    }
    if (solutionValues) {
      cout_doing << "solutionValues[" << offsetRhsValues 
                 << "] = " << solutionValues[offsetRhsValues] << "\n";
    }
    cout_doing << "-------------------------------" << "\n";
  } // end for cell
} // end printValues()
