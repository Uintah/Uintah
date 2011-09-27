/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


#include <CCA/Components/ICE/ExchangeCoefficients.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Util/DebugStream.h>
#include <iostream>
#include <sstream>

using namespace Uintah;
using namespace std;

static DebugStream cout_norm("ICE_NORMAL_COUT", false);  

ExchangeCoefficients::ExchangeCoefficients()
{
  d_heatExchCoeffModel = "constant"; // default
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
    // Pull out the constant Coeff exchange coefficients
    ProblemSpecP exch_ps = ps->findBlock("exchange_properties");
    if (!exch_ps){
      throw ProblemSetupException("Cannot find exchange_properties tag", __FILE__, __LINE__);
    }
    
    //__________________________________
    // variable coefficient models
    exch_ps->get("heatExchangeCoeff",d_heatExchCoeffModel);
    
    if(d_heatExchCoeffModel !="constant" &&
       d_heatExchCoeffModel !="variable" &&
       d_heatExchCoeffModel !="Variable"){
       ostringstream warn;
        warn<<"ERROR\n Heat exchange coefficient model (" << d_heatExchCoeffModel 
            <<") does not exist.\n";
        throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
    }
    
    //__________________________________
    //  constant coefficient model
    ProblemSpecP exch_co_ps = exch_ps->findBlock("exchange_coefficients");

    // momentum
    d_K_mom.clear();
    exch_co_ps->require("momentum",d_K_mom);
    
    // Bullet Proofing
    for (int i = 0; i<(int)d_K_mom.size(); i++) {
      cout_norm << "K_mom = " << d_K_mom[i] << endl;
      if( d_K_mom[i] < 0.0 || d_K_mom[i] > 1e20 ) {
        ostringstream warn;
        warn<<"ERROR\n Momentum exchange coef. is either too big or negative\n";
        throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
      }
    }
    
    // heat
    if(d_heatExchCoeffModel == "constant"){
      d_K_heat.clear();
      exch_co_ps->require("heat",d_K_heat);
    
      // Bullet Proofing
      for (int i = 0; i<(int)d_K_heat.size(); i++) {
        cout_norm << "K_heat = " << d_K_heat[i] << endl;
        if( d_K_heat[i] < 0.0 || d_K_heat[i] > 1e15 ) {
          ostringstream warn;
          warn<<"ERROR\n Heat exchange coef. is either too big or negative\n";
          throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
        }
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
  exch_prop_ps->appendElement("heatExchangeCoeff",d_heatExchCoeffModel);
  
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
