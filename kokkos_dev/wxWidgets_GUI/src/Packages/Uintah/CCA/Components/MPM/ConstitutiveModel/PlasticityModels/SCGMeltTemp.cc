#include "SCGMeltTemp.h"
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Exceptions/InvalidValue.h>
#include <math.h>
#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <sstream>
#include <sgi_stl_warnings_on.h>

using namespace Uintah;
using namespace SCIRun;
using namespace std;

// Construct a melt temp model.  
SCGMeltTemp::SCGMeltTemp(ProblemSpecP& ps )
{
  ps->require("Gamma_0",d_Gamma0);
  ps->require("a",d_a);
  ps->require("T_m0",d_Tm0);
}

// Construct a copy of a melt temp model.  
SCGMeltTemp::SCGMeltTemp(const SCGMeltTemp* mtm)
{
  d_Gamma0 = mtm->d_Gamma0;
  d_a = mtm->d_a;
  d_Tm0 = mtm->d_Tm0;
}

// Destructor of melt temp model.  
SCGMeltTemp::~SCGMeltTemp()
{
}

void SCGMeltTemp::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP temp_ps = ps->appendChild("melting_temp_model",true,4);
  temp_ps->setAttribute("type","scg_Tm");

  temp_ps->appendElement("Gamma_0",d_Gamma0,false,5);
  temp_ps->appendElement("a",d_a,false,5);
  temp_ps->appendElement("T_m0",d_Tm0,false,5);
}

	 
// Compute the melt temp
double 
SCGMeltTemp::computeMeltingTemp(const PlasticityState* state)
{
  double eta = state->density/state->initialDensity;
  double power = 2.0*(d_Gamma0 - d_a - 1.0/3.0);
  double Tm = d_Tm0*exp(2.0*d_a*(1.0 - 1.0/eta))*pow(eta,power);
  
  //cout << " SCG Melting Temp : " << Tm << " eta = " << eta << endl;
  return Tm;
}

