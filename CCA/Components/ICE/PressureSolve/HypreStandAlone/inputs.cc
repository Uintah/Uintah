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
#include "inputs.h"

using namespace std;

double diffusion(const Location& x)
  /*_____________________________________________________________________
    Function diffusion:
    Compute a(x), the scalar diffusion coefficient at a point x in R^d.
    _____________________________________________________________________*/
{
  return 1.0;
}
