#include <Packages/Uintah/CCA/Components/ICE/ExchangeCoefficients.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Core/Util/DebugStream.h>
#include <iostream>
#include <sstream>

using namespace Uintah;
using namespace SCIRun;
using namespace std;

static DebugStream cout_norm("ICE_NORMAL_COUT", false);  

ExchangeCoefficients::ExchangeCoefficients()
{
  d_convective = false;
  d_K_mom.clear();
  d_K_heat.clear();
}

ExchangeCoefficients::~ExchangeCoefficients()
{
}


void ExchangeCoefficients::problemSetup(ProblemSpecP& ps,
                                        SimulationStateP& sharedState)
{
  if(sharedState->getNumMatls() >1){
    //__________________________________
    // Pull out the exchange coefficients
    ProblemSpecP exch_ps = ps->findBlock("exchange_properties");
    if (!exch_ps)
      throw ProblemSetupException("Cannot find exchange_properties tag", __FILE__, __LINE__);

    ProblemSpecP exch_co_ps = exch_ps->findBlock("exchange_coefficients");
    d_K_mom.clear();
    d_K_heat.clear(); 
    exch_co_ps->require("momentum",d_K_mom);
    exch_co_ps->require("heat",d_K_heat);

    for (int i = 0; i<(int)d_K_mom.size(); i++) {
      cout_norm << "K_mom = " << d_K_mom[i] << endl;
      if( d_K_mom[i] < 0.0 || d_K_mom[i] > 1e15 ) {
        ostringstream warn;
        warn<<"ERROR\n Momentum exchange coef. is either too big or negative\n";
        throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
      }
    }
    for (int i = 0; i<(int)d_K_heat.size(); i++) {
      cout_norm << "K_heat = " << d_K_heat[i] << endl;
      if( d_K_heat[i] < 0.0 || d_K_heat[i] > 1e15 ) {
        ostringstream warn;
        warn<<"ERROR\n Heat exchange coef. is either too big or negative\n";
        throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
      }
    }
    //__________________________________
    //  convective heat transfer
    d_convective = false;
    exch_ps->get("do_convective_heat_transfer", d_convective);
    if(d_convective){
      exch_ps->require("convective_fluid",d_conv_fluid_matlindex);
      exch_ps->require("convective_solid",d_conv_solid_matlindex);
    }
  }
}

//______________________________________________________________________
void ExchangeCoefficients::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP exch_prop_ps = ps->appendChild("exchange_properties");
  ProblemSpecP exch_coeff_ps = 
    exch_prop_ps->appendChild("exchange_coefficients");
  
  
  exch_coeff_ps->appendElement("momentum",d_K_mom);
  exch_coeff_ps->appendElement("heat",d_K_heat);

  if (d_convective) {
    exch_coeff_ps->appendElement("do_convective_heat_transfer",d_convective);
    exch_coeff_ps->appendElement("convective_fluid",d_conv_fluid_matlindex);
    exch_coeff_ps->appendElement("convective_solid",d_conv_solid_matlindex);
  }

}

bool ExchangeCoefficients::convective()
{
  return d_convective;
}

int ExchangeCoefficients::conv_fluid_matlindex()
{
  return d_conv_fluid_matlindex;
}

int ExchangeCoefficients::conv_solid_matlindex()
{
  return d_conv_solid_matlindex;
}

vector<double> ExchangeCoefficients::K_mom()
{
  return d_K_mom;
}

vector<double> ExchangeCoefficients::K_heat()
{
  return d_K_heat;
}
