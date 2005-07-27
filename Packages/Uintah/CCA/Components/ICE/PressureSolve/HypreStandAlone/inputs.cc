/*--------------------------------------------------------------------------
 * File: inputs.cc
 *
 * Functions that describe the specific diffusion problem we solve in this
 * code: diffusion coefficient, its harmonic average, RHS for PDE and BC.
 *
 * Revision history:
 * 24-JUL-2005   Oren   Created.
 *--------------------------------------------------------------------------*/

#include "mydriver.h"
#include "Param.h"
#include "inputs.h"
#include "util.h"
#include <vector>

using namespace std;

void
initTest(Param& param)
  /*_____________________________________________________________________
    Function initTest:
    function initTest
    INITTEST  Initialize parameter structure with test information.
    INITTEST sets some PARAM fields that are useful for saving the results.
    Call before starting any runs with this test case.
    List of test cases:
    "linear"
    U = linear function with Dirichlet B.C. on the d-D unit
    square. Diffusion a=1 (Laplace operator). U is smooth.
    "quad1"
    U = quadratic function with Dirichlet B.C. on the d-D unit
    square. Diffusion a=1 (Laplace operator). U is smooth. U
    depends only on x1.
    "quad2"
    U = quadratic function with Dirichlet B.C. on the d-D unit
    square. Diffusion a=1 (Laplace operator). U is smooth.
    "sinsin"
    U = sin(pi*x1)*sin(pi*x2) with Dirichlet B.C. on the d-D unit
    square. Diffusion a=1 (Laplace operator). U is smooth.
    "GaussianSource"
    U = is the solution to Laplace"s equation on the d-D unit square
    with Gaussian right hand side, centered at (0.5,...,0.5) with
    standard deviation of (0.05,...,0.05) on the 2D unit square. 
    Diffusion a=1 (Laplace operator). U is smooth but localized 
    around the source, so at coarse level it is beneficial to locally
    refine around the center of the domain.
    "jump_linear"
    Piecewise constant diffusion coefficient a on the d-D unit
    square. a has a big jump at the hyperplane x1=0.5 (a,u depend only on
    x1; a = aLeft for x1 <= 0.5, a = aRight otherwise). Piecewise linear
    solution U that solves Laplace"s equation with this a.
    "jump_quad"
    Like jump_linear, but with a piecewise quadratic solution U
    that solves Poisson"s equation with RHS = -1, this a, and appropriate
    B.C.
    "diffusion_linear_quad"
    a = 1 + x{1} and u = x{1}^2 (d-D; linear diffusion and smooth
    quadratic solution). Appropriate RHS and Dirichlet BC.
    "diffusion_quad_quad"
    a = 1 + x{1}^2 and u = x{1}^2 (d-D; quadratic diffusion and smooth
    quadratic solution). Appropriate RHS and Dirichlet BC.
    "Lshaped" (2-D only)
    U = r^(2/3)*sin(2*theta/3) is the solution the Laplace"s
    equation with Dirichlet B.C. on the L-shaped domain [0,1]^2 \
    [0.5,1]^2.Diffusion a=1 (Laplace operator). This is a
    re-entrant corner problem where U is singular.
    _____________________________________________________________________*/
{

  param.supportedDims.clear();

  switch (param.problemType) {
  
  case Param::Linear:
    /* u is a linear function (d-D) */
    {
      param.longTitle     = "Linear Solution";
      param.supportedDims.push_back(-1); // Works in any dimension
      break;
    }

  case Param::Quad1:
    {
      param.longTitle     = "Quadratic Solution $u=u(x_1)$";
      param.supportedDims.push_back(-1); // Works in any dimension
      break;
    }
    
  case Param::Quad2:
    {
      param.longTitle     = "Quadratic Solution";
      param.supportedDims.push_back(-1); // Works in any dimension
      break;
    }
    
  case Param::SinSin:
    {
      param.longTitle     = "Smooth Solution";
      param.supportedDims.push_back(-1); // Works in any dimension
      break;
    }
    
  case Param::GaussianSource:
    {
      param.longTitle     = "Localized Gaussian Source";
      param.supportedDims.push_back(-1); // Works in any dimension
      break;
    }
          
  case Param::JumpLinear:
    {
      param.longTitle     =
        "Piecewise Constant Diffusion, " \
        "Piecewise Linear Solution";
      param.supportedDims.push_back(-1); // Works in any dimension
      break;
    }
    
    case Param::JumpQuad:
      {
        param.longTitle     = 
          "Piecewise Constant Diffusion, " \
          "Piecewise Quadratic Solution";
        param.supportedDims.push_back(-1); // Works in any dimension
        break;
      }
      
  case Param::DiffusionQuadLinear:
    {
      param.longTitle     = "Linear Diffusion, Quadratic Solution";
      param.supportedDims.push_back(-1); // Works in any dimension
        break;
      }
    
  case Param::DiffusionQuadQuad:
    {
      param.longTitle     = "Quadratic Diffusion, Quadratic Solution";
      param.supportedDims.push_back(-1); // Works in any dimension
      break;
    }
    
  case Param::LShaped:
    {
      param.longTitle     = "L-Shaped Reentrant Corner";
      param.supportedDims.push_back(2);    // Works only 2-D
      break;
    }
    
  default:
    {
      Print("\n\nError, unknown problem type in diffusion().\n");
      clean();
      exit(1);
    }
  } // end switch (param.problemType)
} // end initTest

double
diffusion(const int problemType,
          const Location& x)
  /*_____________________________________________________________________
    Function diffusion:
    Compute a(x), the scalar diffusion coefficient at a point x in R^d.
    problemType is part of the Param parameters structure and determines
    which problem we're solving.
    _____________________________________________________________________*/
{
  switch (problemType) {
    // Problems with a=1 (Laplace operator)
  case Param::Linear:
  case Param::Quad1:
  case Param::Quad2:
  case Param::SinSin:
  case Param::GaussianSource:
  case Param::LShaped:
    {
      return 1.0;
    }
    
  case Param::JumpLinear:
    // Piecewise constant diffusion coefficient with a big jump at x1=0.5.
    {
      double x0          = 0.5;
      double aLeft       = 1.0;
      double aRight      = 1.0e+6;
      if (x[0] < x0) return aLeft;
      return aRight;
    }
    
  case Param::JumpQuad:
    // Piecewise constant diffusion coefficient with a big jump at x1=0.5.
    {
      double x0          = 0.53;
      double aLeft       = 10.0;
      double aRight      = 1.0e+6;
      if (x[0] < x0) return aLeft;
      return aRight;
    }
    
  case Param::DiffusionQuadLinear:
    {
      return 1 + x[0];
      break;
    }
  case Param::DiffusionQuadQuad:
    {
      return 1 + x[0]*x[0];
    }
  default:
    {
      Print("\n\nError, unknown problem type in diffusion().\n");
      clean();
      exit(1);
    }
  } // end switch (param.problemType)
}
