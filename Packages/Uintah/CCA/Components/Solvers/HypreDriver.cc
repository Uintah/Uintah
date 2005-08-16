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
#include <Packages/Uintah/CCA/Components/Solvers/HypreGenericSolver.h>
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
    class HypreDriver implementation common to all variable types
    and all Hypre interfaces
    _____________________________________________________________________*/

  HypreDriver::~HypreDriver(void)
  {
#if 0
    if (_activeInterface & HypreSStruct) {
      cerr << "Destroying SStruct matrix, RHS, solution objects" << "\n";
      HYPRE_SStructMatrixDestroy(_A_SStruct);
      HYPRE_SStructVectorDestroy(_b_SStruct);
      HYPRE_SStructVectorDestroy(_x_SStruct);
      cerr << "Destroying graph objects" << "\n";
      HYPRE_SStructGraphDestroy(_graph_SStruct);
    }
    if (_activeInterface & HypreParCSR) {
      cerr << "Destroying ParCSR matrix, RHS, solution objects" << "\n";
      HYPRE_ParCSRMatrixDestroy(_A_Par);
      HYPRE_ParVectorDestroy(_b_Par);
      HYPRE_ParVectorDestroy(_x_Par);
    }
#endif
  }

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
  
  TypeDescription::Type TypeTemplate2Enum(const SFCXTypes& t)
  {
    return TypeDescription::SFCXVariable;
  }

  TypeDescription::Type TypeTemplate2Enum(const SFCYTypes& t)
  {
    return TypeDescription::SFCYVariable;
  }

  TypeDescription::Type TypeTemplate2Enum(const SFCZTypes& t)
  {
    return TypeDescription::SFCZVariable;
  }

  TypeDescription::Type TypeTemplate2Enum(const NCTypes& t)
  {
    return TypeDescription::NCVariable;
  }

  TypeDescription::Type TypeTemplate2Enum(const CCTypes& t)
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
           which_guess_dw, params);
      }
#if 0
    case HypreSStruct:
      {
        return new HypreDriverSStruct
          (level, matlset, A, which_A_dw,
           x, modifies_x, b, which_b_dw, guess, 
           which_guess_dw, params);
      }
#endif
    default:
      throw InternalError("Unsupported Hypre Interface: "+interface,
                          __FILE__, __LINE__);
    } // end switch (interface)
  }

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

  double harmonicAvg(const Point& x,
                     const Point& y,
                     const Point& z,
                     const double& Ax,
                     const double& Ay)
    /*_____________________________________________________________________
      Function harmonicAvg~: 
      Harmonic average of the diffusion coefficient.
      A = harmonicAvg(X,Y,Z) returns the harmonic average of the
      diffusion coefficient a(T) (T in R^D) along the line connecting
      the points X,Y in R^D. That is, A = 1/(integral_0^1
      1/a(t1(s),...,tD(s)) ds), where td(s) = x{d} + s*(y{d} -
      x{d})/norm(y-x) is the arclength parameterization of the
      d-coordinate of the line x-y, d = 1...D.  We assume that A is
      piecewise constant with jump at Z (X,Y are normally cell centers
      and Z at the cell face). X,Y,Z are Dx1 location arrays.  In
      general, A can be analytically computed for the specific cases we
      consider; in general, use some simple quadrature formula for A
      from discrete a-values. This can be implemented by the derived
      test cases from Param.

      ### NOTE: ### If we use a different
      refinement ratio in different dimensions, near the interface we
      may need to compute A along lines X-Y that cross more than one
      cell boundary. This is currently ignored and we assume all lines
      cut one cell interface only.
      _____________________________________________________________________*/

  {
    const int numDims = 3;
    /* Compute distances x-y and x-z */
    double dxy = 0.0, dxz = 0.0;
    for (int d = 0; d < numDims; d++) {
      dxy += pow(fabs(y(d) - x(d)),2.0);
      dxz += pow(fabs(z(d) - x(d)),2.0);
    }
    double K = sqrt(dxz/dxy);
    return (Ax*Ay)/((1-K)*Ax + K*Ay);
  }

} // end namespace Uintah
