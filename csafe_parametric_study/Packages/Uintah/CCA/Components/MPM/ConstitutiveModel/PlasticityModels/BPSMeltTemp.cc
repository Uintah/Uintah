#include "BPSMeltTemp.h"
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
BPSMeltTemp::BPSMeltTemp(ProblemSpecP& ps )
{
  ps->require("B0", d_B0);       
  ps->require("dB_dp0", d_dB_dp0);   
  ps->require("G0", d_G0);       
  ps->require("dG_dp0", d_dG_dp0);   
  ps->require("kappa", d_kappa);  
  ps->require("z", d_z);        
  ps->require("b2rhoTm", d_b2rhoTm);
  ps->require("alpha", d_alpha);
  ps->require("lambda", d_lambda);
  ps->require("a", d_a);        
  ps->require("v_ws_a3_factor", d_factor);        
  ps->require("Boltzmann_Constant", d_kb);        
}

// Construct a copy of a melt temp model.  
BPSMeltTemp::BPSMeltTemp(const BPSMeltTemp* mtm)
{
  d_B0 = mtm->d_B0;
  d_dB_dp0 = mtm->d_dB_dp0;
  d_G0 = mtm->d_G0;
  d_dG_dp0 = mtm->d_dG_dp0;
  d_kappa = mtm->d_kappa;
  d_z = mtm->d_z;
  d_b2rhoTm = mtm->d_b2rhoTm;
  d_alpha = mtm->d_alpha;
  d_lambda = mtm->d_lambda;
  d_a = mtm->d_a;
  d_factor = mtm->d_factor;
  d_kb = mtm->d_kb;
}

// Destructor of melt temp model.  
BPSMeltTemp::~BPSMeltTemp()
{
}
	 
void BPSMeltTemp::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP temp_ps = ps->appendChild("melting_temp_model");
  temp_ps->setAttribute("type","bps_Tm");

  temp_ps->appendElement("B0", d_B0);       
  temp_ps->appendElement("dB_dp0", d_dB_dp0);   
  temp_ps->appendElement("G0", d_G0);       
  temp_ps->appendElement("dG_dp0", d_dG_dp0);   
  temp_ps->appendElement("kappa", d_kappa);  
  temp_ps->appendElement("z", d_z);        
  temp_ps->appendElement("b2rhoTm", d_b2rhoTm);
  temp_ps->appendElement("alpha", d_alpha);
  temp_ps->appendElement("lambda", d_lambda);
  temp_ps->appendElement("a", d_a);        
  temp_ps->appendElement("v_ws_a3_factor", d_factor);        
  temp_ps->appendElement("Boltzmann_Constant", d_kb);        
}

// Compute the melt temp
double 
BPSMeltTemp::computeMeltingTemp(const PlasticityState* state)
{
  // Calculate the melt temperature at zero pressure and 300 K
  double p = -state->pressure;
  double vws = pow(d_a, 3.0)/d_factor;
  double kbTm = (d_kappa*d_lambda*vws*d_G0)/(8.0*M_PI*log(d_z-1.0))*
                log((d_alpha*d_alpha)/(4.0*d_b2rhoTm));
  double Tm0 = kbTm/d_kb;

  // Calculate the bulk and shear factors
  double Bfac = 1.0 + d_dB_dp0/d_B0*p;
  double Gfac = d_dG_dp0/d_G0*p;

  // Calculate the compression using the Murnaghan EOS
  double eta = pow(Bfac, (1.0/d_dB_dp0));
  
  // Calculate Tm at pressure p
  double Tm = Tm0/eta*(1.0 + Gfac/pow(eta,(1.0/3.0)));

  //cout << " BPS Melting Temp : " << Tm << " eta = " << eta << endl;
  //cout << "     p = " << p << " vws = " << vws << " kbTm = " << kbTm << endl;
  //cout << "     Bfac = " << Bfac << " Gfac = " << Gfac << endl;
  return Tm;
}

