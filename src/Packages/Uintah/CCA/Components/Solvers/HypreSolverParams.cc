//--------------------------------------------------------------------------
// File: HypreSolverParams.cc
//
// Implementation of the parameter structure of HypreSolverAMR.
//--------------------------------------------------------------------------
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

using namespace Uintah;
//__________________________________
//  To turn on normal output
//  setenv SCI_DEBUG "HYPRE_DOING_COUT:+"

static DebugStream cout_doing("HYPRE_DOING_COUT", false);

namespace Uintah {

  double HypreSolverParams::harmonicAvg(const Location& x,
                                        const Location& y,
                                        const Location& z) const
    /*_____________________________________________________________________
      Function harmonicAvg: 
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
    double Ax = diffusion(x);
    double Ay = diffusion(y);
    /* Compute distances x-y and x-z */
    double dxy = 0.0, dxz = 0.0;
    for (Counter d = 0; d < numDims; d++) {
      dxy += pow(fabs(y[d] - x[d]),2.0);
      dxz += pow(fabs(z[d] - x[d]),2.0);
    }
    double K = sqrt(dxz/dxy);
    return (Ax*Ay)/((1-K)*Ax + K*Ay);
  }

} // end namespace Uintah
