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

#include <CCA/Components/MPM/ConstitutiveModel/PlasticityModels/ZAPolymerFlow.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <cmath>

using namespace std;
using namespace Uintah;
//______________________________________________________________________

ZAPolymerFlow::ZAPolymerFlow(ProblemSpecP& ps)
{
  ps->require("sigma_g",  d_CM.sigma_g);   
  ps->require("B_pa",     d_CM.B_pa);      
  ps->require("B_pb",     d_CM.B_pb);      
  ps->require("B_pn",     d_CM.B_pn);      
  ps->require("beta_0",   d_CM.beta_0);    
  ps->require("beta_1",   d_CM.beta_1);    
  ps->require("T_0",      d_CM.T_0);       
  ps->require("B_0pa",    d_CM.B_0pa);     
  ps->require("B_0pb",    d_CM.B_0pb);     
  ps->require("B_0pn",    d_CM.B_0pn);     
  ps->require("omega_a",  d_CM.omega_a);   
  ps->require("omega_b",  d_CM.omega_b);   
  ps->require("omega_p",  d_CM.omega_p);   
}
 //__________________________________
 //        
ZAPolymerFlow::ZAPolymerFlow(const ZAPolymerFlow* cm)
{
  d_CM.sigma_g  =  cm->d_CM.sigma_g;
  d_CM.B_pa     =  cm->d_CM.B_pa;
  d_CM.B_pb     =  cm->d_CM.B_pb;
  d_CM.B_pn     =  cm->d_CM.B_pn;
  d_CM.beta_0   =  cm->d_CM.beta_0;
  d_CM.beta_1   =  cm->d_CM.beta_1;
  d_CM.T_0      =  cm->d_CM.T_0;
  d_CM.B_0pa    =  cm->d_CM.B_0pa;
  d_CM.B_0pb    =  cm->d_CM.B_0pb;
  d_CM.B_0pn    =  cm->d_CM.B_0pn;
  d_CM.omega_a  =  cm->d_CM.omega_a;  
  d_CM.omega_b  =  cm->d_CM.omega_b;  
  d_CM.omega_p  =  cm->d_CM.omega_p;
  d_CM.alpha_0  =  cm->d_CM.alpha_0;
  d_CM.alpha_1  =  cm->d_CM.alpha_1;
}
         
ZAPolymerFlow::~ZAPolymerFlow()
{
}
//__________________________________
//
void ZAPolymerFlow::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP flow_ps = ps->appendChild("flow_model");
  flow_ps->setAttribute("type","zerilli_armstrong_polymer");

  flow_ps->appendElement("sigma_g",  d_CM.sigma_g);   
  flow_ps->appendElement("B_pa",     d_CM.B_pa);      
  flow_ps->appendElement("B_pb",     d_CM.B_pb);      
  flow_ps->appendElement("B_pn",     d_CM.B_pn);      
  flow_ps->appendElement("beta_0",   d_CM.beta_0);    
  flow_ps->appendElement("beta_1",   d_CM.beta_1);    
  flow_ps->appendElement("T_0",      d_CM.T_0);       
  flow_ps->appendElement("B_0pa",    d_CM.B_0pa);     
  flow_ps->appendElement("B_0pb",    d_CM.B_0pb);     
  flow_ps->appendElement("B_0pn",    d_CM.B_0pn);     
  flow_ps->appendElement("omega_a",  d_CM.omega_a);   
  flow_ps->appendElement("omega_b",  d_CM.omega_b);   
  flow_ps->appendElement("omega_p",  d_CM.omega_p);   

}

//______________________________________________________________________
//     Reference & equation number????
double 
ZAPolymerFlow::computeFlowStress(const PlasticityState* state,
                             const double& ,
                             const double& ,
                             const MPMMaterial* ,
                             const particleIndex idx)
{
  double epdot = state->plasticStrainRate;
  double ep    = state->plasticStrain;
  double T     = state->temperature;
  double P     = -state->pressure;
  
  // Clamps & bulletproofing
  epdot = (epdot == 0.0) ? 1.0e-10 : epdot;
  P = (P <= 0.0) ? 0.0 : P;
  
  if( ep < 0.0 || T < 0.0){
    throw InternalError("ZAPolymerFlow::computeFlowStress a negative temperature "
                        "or equivalent plastic strain has been detected",__FILE__,__LINE__);
  }

  // Intermediate quantities
  double ln_epdot = log(epdot);
  double alpha   = d_CM.alpha_0 - d_CM.alpha_1 * ln_epdot;
  double beta    = d_CM.beta_0  - d_CM.beta_1  * ln_epdot;
  
  double B       = d_CM.B_pa  * pow( ( 1.0 + d_CM.B_pb  * sqrt(P) ), d_CM.B_pn ); 
  double B_0     = d_CM.B_0pa * pow( ( 1.0 + d_CM.B_0pb * sqrt(P) ), d_CM.B_0pn );
  double omega   = d_CM.omega_a + d_CM.omega_b * ln_epdot + d_CM.omega_p * sqrt(P);
  
  double T_T0    = T - d_CM.T_0;
 
 
  double sigma_y = d_CM.sigma_g
                 + B * exp( -beta * T_T0 )
                 + B_0 * sqrt(omega * ep)* exp( -alpha * T_T0 );
  
  if (isnan(sigma_y)) {
    cout << "WARNING::ZAPolymerFlow::computeFlowStress:: idx = " << idx << " epdot = " << epdot
         << " ep = " << ep << " T = " << T << endl;
    cout << " P = " << P << " d_CM.sigma_g = " << d_CM.sigma_g
          << " alpha = " << alpha << " beta = " << beta
          << " B = " << B << " B_0 " << B_0 << " omega " << omega 
          << " sigma_y = " << sigma_y << endl;
  }

  return sigma_y;
}

//______________________________________________________________________
//  
double
ZAPolymerFlow::evalDerivativeWRTPlasticStrain(const PlasticityState* state,
                                          const particleIndex )
{
  // Get the state data
  double ep    = state->plasticStrain;
  double epdot = state->plasticStrainRate;
  double T     = state->temperature;
  double P     = -state->pressure;
  
  // Clamps & bulletproofing
  epdot = (epdot == 0.0) ? 1.0e-10 : epdot;
  ep = (ep == 0.0) ? 1.0e-8 : ep;
  P = (P <= 0.0) ? 0.0 : P;
  
  if( ep < 0.0 || T < 0.0){
    throw InternalError("ZAPolymerFlow::evalDerivativeWRTPlasticStrain a negative temperature "
                        "or equivalent plastic strain has been detected",__FILE__,__LINE__);
  }
  
  double ln_epdot = log(epdot);
  double alpha = d_CM.alpha_0 - d_CM.alpha_1 * ln_epdot;
  double omega = d_CM.omega_a + d_CM.omega_b * ln_epdot + d_CM.omega_p * sqrt(P);
  double B_0   = d_CM.B_0pa * pow(  ( 1.0 + d_CM.B_0pb * sqrt(P) ), d_CM.B_0pn );
  
  double T_T0  = T - d_CM.T_0;  

  double deriv = 0.0;

  if( omega > 0.0){
    deriv = 0.5 * B_0 * omega * exp(-alpha * T_T0)/sqrt(omega * ep); 
  }
  return deriv;
}

//______________________________________________________________________
//
double
ZAPolymerFlow::computeShearModulus(const PlasticityState* state)
{
  return state->shearModulus;
}


//______________________________________________________________________
//
double
ZAPolymerFlow::evalDerivativeWRTStrainRate(const PlasticityState* state,
                                       const particleIndex )
{
  // Get the state data
  double ep    = state->plasticStrain;
  double epdot = state->plasticStrainRate;
  double T     = state->temperature;
  double P     = -state->pressure;
  
  // Clamps & bulletproofing
  epdot = (epdot == 0.0) ? 1.0e-10 : epdot;
  ep = (ep == 0.0) ? 1.0e-8 : ep;
  P = (P <= 0.0) ? 0.0 : P;
  
  if( ep < 0.0 || T < 0.0){
    throw InternalError("ZAPolymerFlow::evalDerivativeWRTStrainRate a negative temperature "
                        "or equivalent plastic strain has been detected",__FILE__,__LINE__);
  }
  
  double ln_epdot = log(epdot);
  double alpha = d_CM.alpha_0 - d_CM.alpha_1 * ln_epdot;
  double beta  = d_CM.beta_0  - d_CM.beta_1  * ln_epdot;
  double omega = d_CM.omega_a + d_CM.omega_b * ln_epdot + d_CM.omega_p * sqrt(P);
    
  double B     = d_CM.B_pa  * pow(  ( 1.0 + d_CM.B_pb  * sqrt(P) ), d_CM.B_pn );
  double B_0   = d_CM.B_0pa * pow(  ( 1.0 + d_CM.B_0pb * sqrt(P) ), d_CM.B_0pn );
  double T_T0  = T - d_CM.T_0;
  
  double term1 = B   * d_CM.beta_1 *  exp(-beta * T_T0);
  double term2 = B_0 * d_CM.alpha_1 * sqrt(omega * ep)  * exp(-alpha * T_T0);
  
  double deriv = (term1+term2) * T_T0 /epdot;

  return deriv;
}


