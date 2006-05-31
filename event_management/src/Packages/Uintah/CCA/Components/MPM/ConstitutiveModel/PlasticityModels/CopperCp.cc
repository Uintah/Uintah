#include "CopperCp.h"
#include <math.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>

using namespace Uintah;
	 
// Construct a specific heat model.  
CopperCp::CopperCp(ProblemSpecP& ps)
{
  d_T0 = 270.0;
  ps->get("T_transition",d_T0);
  d_A0 = 0.0000416;
  ps->get("A_LowT",d_A0);
  d_B0 = 0.027;
  ps->get("B_LowT",d_B0);
  d_C0 = 6.21;
  ps->get("C_LowT",d_C0);
  d_D0 = 142.6;
  ps->get("D_LowT",d_D0);
  d_A1 = 0.1009;
  ps->get("A_HighT",d_A1);
  d_B1 = 358.4;
  ps->get("B_HighT",d_B1);
}

// Construct a copy of a specific heat model.  
CopperCp::CopperCp(const CopperCp* ccp)
{
  d_T0 = ccp->d_T0;
  d_A0 = ccp->d_A0;
  d_B0 = ccp->d_B0;
  d_C0 = ccp->d_C0;
  d_D0 = ccp->d_D0;
  d_A1 = ccp->d_A1;
  d_B1 = ccp->d_B1;
}

// Destructor of specific heat model.  
CopperCp::~CopperCp()
{
}
	 
void CopperCp::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP cm_ps = ps->appendChild("specific_heat_model");
  cm_ps->setAttribute("type","copper_Cp");

  cm_ps->appendElement("T_transition",d_T0);
  cm_ps->appendElement("A_LowT",d_A0);
  cm_ps->appendElement("B_LowT",d_B0);
  cm_ps->appendElement("C_LowT",d_C0);
  cm_ps->appendElement("D_LowT",d_D0);
  cm_ps->appendElement("A_HighT",d_A1);
  cm_ps->appendElement("B_HighT",d_B1);
}

// Compute the specific heat
double 
CopperCp::computeSpecificHeat(const PlasticityState* state)
{
  double T = state->temperature;
  double Cp = 414.0;
  if (T < d_T0) {
    Cp = d_A0*(T*T*T) - d_B0*(T*T) + d_C0*T - d_D0;
  } else {
    Cp = d_A1*T + d_B1;
  }
  return Cp;
}

