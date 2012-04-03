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


#ifdef __APPLE__
// This is a hack.  gcc 3.3 #undefs isnan in the cmath header, which
// make the isnan function not work.  This define makes the cmath header
// not get included since we do not need it anyway.
#  define _CPP_CMATH
#endif

#include <CCA/Components/MPM/ConstitutiveModel/PlasticityModels/JohnsonCookPlastic.h>
#include <cmath>


using namespace std;
using namespace Uintah;

JohnsonCookPlastic::JohnsonCookPlastic(ProblemSpecP& ps)
{
  ps->require("A",d_CM.A);
  ps->require("B",d_CM.B);
  ps->require("C",d_CM.C);
  ps->require("n",d_CM.n);
  ps->require("m",d_CM.m);
  d_CM.epdot_0 = 1.0;
  ps->get("epdot_0", d_CM.epdot_0);
  d_CM.TRoom = 298;
  ps->get("T_r",d_CM.TRoom);
  d_CM.TMelt = 1793;
  ps->get("T_m",d_CM.TMelt);
}
         
JohnsonCookPlastic::JohnsonCookPlastic(const JohnsonCookPlastic* cm)
{
  d_CM.A = cm->d_CM.A;
  d_CM.B = cm->d_CM.B;
  d_CM.C = cm->d_CM.C;
  d_CM.n = cm->d_CM.n;
  d_CM.m = cm->d_CM.m;
  d_CM.epdot_0 = cm->d_CM.epdot_0;
  d_CM.TRoom = cm->d_CM.TRoom;
  d_CM.TMelt = cm->d_CM.TMelt;
}
         
JohnsonCookPlastic::~JohnsonCookPlastic()
{
}

void JohnsonCookPlastic::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP plastic_ps = ps->appendChild("plasticity_model");
  plastic_ps->setAttribute("type","johnson_cook");

  plastic_ps->appendElement("A",d_CM.A);
  plastic_ps->appendElement("B",d_CM.B);
  plastic_ps->appendElement("C",d_CM.C);
  plastic_ps->appendElement("n",d_CM.n);
  plastic_ps->appendElement("m",d_CM.m);
  plastic_ps->appendElement("epdot_0", d_CM.epdot_0);
  plastic_ps->appendElement("T_r",d_CM.TRoom);
  plastic_ps->appendElement("T_m",d_CM.TMelt);
}

         
void 
JohnsonCookPlastic::addInitialComputesAndRequires(Task* ,
                                           const MPMMaterial* ,
                                           const PatchSet*)
{
}

void 
JohnsonCookPlastic::addComputesAndRequires(Task* ,
                                    const MPMMaterial* ,
                                    const PatchSet*)
{
}

void 
JohnsonCookPlastic::addComputesAndRequires(Task* task,
                                   const MPMMaterial* matl,
                                   const PatchSet*,
                                   bool /*recurse*/,
                                   bool /*SchedParent*/)
{
}

void 
JohnsonCookPlastic::addParticleState(std::vector<const VarLabel*>& ,
                                     std::vector<const VarLabel*>& )
{
}

void 
JohnsonCookPlastic::allocateCMDataAddRequires(Task* ,
                                              const MPMMaterial* ,
                                              const PatchSet* ,
                                              MPMLabel* )
{
}

void JohnsonCookPlastic::allocateCMDataAdd(DataWarehouse* ,
                                           ParticleSubset* ,
                                           map<const VarLabel*, 
                                           ParticleVariableBase*>* ,
                                           ParticleSubset* ,
                                           DataWarehouse* )
{
}

void 
JohnsonCookPlastic::initializeInternalVars(ParticleSubset* ,
                                           DataWarehouse* )
{
}

void 
JohnsonCookPlastic::getInternalVars(ParticleSubset* ,
                                    DataWarehouse* ) 
{
}

void 
JohnsonCookPlastic::allocateAndPutInternalVars(ParticleSubset* ,
                                               DataWarehouse* ) 
{
}

void 
JohnsonCookPlastic::allocateAndPutRigid(ParticleSubset* ,
                                        DataWarehouse* ) 
{
}

void
JohnsonCookPlastic::updateElastic(const particleIndex )
{
}

void
JohnsonCookPlastic::updatePlastic(const particleIndex , const double& )
{
}

double 
JohnsonCookPlastic::computeFlowStress(const PlasticityState* state,
                                      const double& ,
                                      const double& ,
                                      const MPMMaterial* matl,
                                      const particleIndex idx )
{
  //double epdot = state->plasticStrainRate/d_CM.epdot_0;
  double epdot = state->strainRate/d_CM.epdot_0;
  double ep = state->plasticStrain;
  double T = state->temperature;
  //double Tr = matl->getRoomTemperature();
  //double Tm = state->meltingTemp;
  double Tr = d_CM.TRoom;
  double Tm = d_CM.TMelt;

  double strainPart = d_CM.A + d_CM.B*pow(ep,d_CM.n);
  double strainRatePart = 1.0;
  if (epdot < 1.0){ 
    strainRatePart = pow((1.0 + epdot),d_CM.C);
  }
  else{
    strainRatePart = 1.0 + d_CM.C*log(epdot);
  }
  double m = d_CM.m;
  double Tstar = (T > Tm) ? 1.0 : ((T-Tr)/(Tm-Tr)); 
  double tempPart = (Tstar < 0.0) ? (1.0 - Tstar) : (1.0-pow(Tstar,m));
  double sigy = strainPart*strainRatePart*tempPart;
  if (isnan(sigy)) {
    cout << "**ERROR** JohnsonCook: sig_y == nan " 
         << " Particle = " << idx << " epdot = " << epdot
         << " ep = " << ep << " T = " << T 
         << " strainPart = " << strainPart 
         << " strainRatePart = " << strainRatePart 
         << " Tstar = " << Tstar 
         << " tempPart = " << tempPart << endl; 
  }
  return sigy;
}

double 
JohnsonCookPlastic::computeEpdot(const PlasticityState* state,
                                 const double& ,
                                 const double& ,
                                 const MPMMaterial* matl,
                                 const particleIndex )
{
  // All quantities should be at the beginning of the 
  // time step
  double tau = state->yieldStress;
  double ep = state->plasticStrain;
  double T = state->temperature;
  //double Tr = matl->getRoomTemperature();
  //double Tm = state->meltingTemp;
  double Tr = d_CM.TRoom;
  double Tm = d_CM.TMelt;

  double strainPart = d_CM.A + d_CM.B*pow(ep,d_CM.n);
  d_CM.TRoom = Tr;  d_CM.TMelt = Tm;
  double m = d_CM.m;
  double Tstar = (T > Tm) ? 1.0 : ((T-Tr)/(Tm-Tr)); 
  double tempPart = (Tstar < 0.0) ? (1.0 - Tstar) : (1.0-pow(Tstar,m));

  double fac1 = tau/(strainPart*tempPart);
  double fac2 = (1.0/d_CM.C)*(fac1-1.0);
  double epdot = exp(fac2)*d_CM.epdot_0;
  if (isnan(epdot)) {
    cout << "**ERROR** JohnsonCook: epdot == nan " << endl; 
  }
  return epdot;
}
 

void 
JohnsonCookPlastic::computeTangentModulus(const Matrix3& stress,
                                          const PlasticityState* ,
                                          const double& ,
                                          const MPMMaterial* ,
                                          const particleIndex ,
                                          TangentModulusTensor& ,
                                          TangentModulusTensor& )
{
  throw InternalError("Empty Function: JohnsonCookPlastic::computeTangentModulus", __FILE__, __LINE__);
}

void
JohnsonCookPlastic::evalDerivativeWRTScalarVars(const PlasticityState* state,
                                                const particleIndex idx,
                                                Vector& derivs)
{
  derivs[0] = evalDerivativeWRTStrainRate(state, idx);
  derivs[1] = evalDerivativeWRTTemperature(state, idx);
  derivs[2] = evalDerivativeWRTPlasticStrain(state, idx);
}


double
JohnsonCookPlastic::evalDerivativeWRTPlasticStrain(const PlasticityState* state,
                                                   const particleIndex )
{
  // Get the state data
  //double epdot = state->plasticStrainRate/d_CM.epdot_0;
  double epdot = state->strainRate/d_CM.epdot_0;
  double ep = state->plasticStrain;
  double T = state->temperature;
  //double Tm = state->meltingTemp;
  double Tr = d_CM.TRoom;
  double Tm = d_CM.TMelt;

  // Calculate strain rate part
  double strainRatePart = (epdot < 1.0) ? 
          (pow((1.0 + epdot),d_CM.C)) : (1.0 + d_CM.C*log(epdot));

  // Calculate temperature part
  double m = d_CM.m;
  double Tstar = (T > Tm) ? 1.0 : (T-Tr)/(Tm-Tr);
  double tempPart = (Tstar < 0.0) ? (1.0 - Tstar) : (1.0-pow(Tstar,m));

  double D = strainRatePart*tempPart;

  double deriv =  (ep > 0.0) ?  (d_CM.B*d_CM.n*D*pow(ep,d_CM.n-1)) : 0.0;
  if (isnan(deriv)) {
    cout << "**ERROR** JohnsonCook: dsig/dep == nan " << endl; 
  }
  return deriv;
}

///////////////////////////////////////////////////////////////////////////
/*  Compute the shear modulus. */
///////////////////////////////////////////////////////////////////////////
double
JohnsonCookPlastic::computeShearModulus(const PlasticityState* state)
{
  return state->shearModulus;
}

///////////////////////////////////////////////////////////////////////////
/* Compute the melting temperature */
///////////////////////////////////////////////////////////////////////////
double
JohnsonCookPlastic::computeMeltingTemp(const PlasticityState* state)
{
  return state->meltingTemp;
}

double
JohnsonCookPlastic::evalDerivativeWRTTemperature(const PlasticityState* state,
                                                 const particleIndex )
{
  // Get the state data
  //double epdot = state->plasticStrainRate/d_CM.epdot_0;
  double epdot = state->strainRate/d_CM.epdot_0;
  double ep = state->plasticStrain;
  double T = state->temperature;
  //double Tm = state->meltingTemp;
  double Tr = d_CM.TRoom;
  double Tm = d_CM.TMelt;

  // Calculate strain part
  double strainPart = d_CM.A + d_CM.B*pow(ep,d_CM.n);

  // Calculate strain rate part
  double strainRatePart = 1.0;
  if (epdot < 1.0) strainRatePart = pow((1.0 + epdot),d_CM.C);
  else strainRatePart = 1.0 + d_CM.C*log(epdot);

  // Calculate temperature part
  double m = d_CM.m;
  double Tstar = (T > Tm) ? 1.0 : (T-Tr)/(Tm-Tr);

  double F = strainPart*strainRatePart;
  double deriv = (T < Tr) ? -F/(Tm-Tr) : -m*F*pow(Tstar,m-1)/(Tm-Tr);
  if (isnan(deriv)) {
    cout << "**ERROR** JohnsonCook: dsig/dT == nan " << endl; 
  }
  return deriv;
}

double
JohnsonCookPlastic::evalDerivativeWRTStrainRate(const PlasticityState* state,
                                                const particleIndex )
{
  // Get the state data
  //double epdot = state->plasticStrainRate/d_CM.epdot_0;
  double epdot = state->strainRate/d_CM.epdot_0;
  double ep = state->plasticStrain;
  double T = state->temperature;
  //double Tm = state->meltingTemp;
  double Tr = d_CM.TRoom;
  double Tm = d_CM.TMelt;

  // Calculate strain part
  double strainPart = d_CM.A + d_CM.B*pow(ep,d_CM.n);

  // Calculate temperature part
  double m = d_CM.m;
  double Tstar = (T > Tm) ? 1.0 : (T-Tr)/(Tm-Tr);
  double tempPart = (Tstar < 0.0) ? (1.0 - Tstar) : (1.0-pow(Tstar,m));

  double E = strainPart*tempPart;

  double deriv = 0.0;
  if (epdot < 1.0) 
    deriv = E*d_CM.C*pow((1.0 + epdot),(d_CM.C-1.0));
  else
    deriv = E*d_CM.C/epdot;
  if (isnan(deriv)) {
    cout << "**ERROR** JohnsonCook: dsig/depdot == nan " << endl; 
  }
  return deriv;

}


