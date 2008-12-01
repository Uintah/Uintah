
#include "MPMEquationOfState.h"

using namespace Uintah;

MPMEquationOfState::MPMEquationOfState()
{
}

MPMEquationOfState::~MPMEquationOfState()
{
}

// Calculate rate of temperature change due to compression/expansion
double
MPMEquationOfState::computeIsentropicTemperatureRate(const double T,
                                                     const double rho_0,
                                                     const double rho_cur,
                                                     const double Dtrace)
{
  double dTdt = 0.;
  return dTdt;
}
