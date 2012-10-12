/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

//  ViscoElasticDamage.cc 
//  class ConstitutiveModel ConstitutiveModel data type -- 3D - 
//  holds ConstitutiveModel
//  information for the FLIP technique:
//    This constitutive model is based on theorem of Continuum Damage 
//    Mechanics (CDM) with following assumptions:
//      i.   Standard solid type of viscoelastic behavior with linear rate
//           equation.
//      ii.  Free energy equation is uncoupled as volumetric and deviatoric
//           parts and the equation is the same as Neo-Hookean model (refer:
//           CompNeoHook.h and CompNeoHook.cc).
//      iii. The damage mechanism is associated with maximum distortional
//           energy and is independent of hydrostatic pressure.
//      iv.  The damage criterion is only function of equivalent strain. 
//      v.   The damage process character function has exponential form.
//    Material property constants:
//      Elasticity: Young's modulus, Poisson's ratio;
//      Damage parameters: alpha[0, infinity), beta[0, 1].
//      Visco properties: tau[0,infinity) -- relaxation time,
//                        gamma[0,1) -- stiffness ratio.
//  Reference: "ON A FULLY THREE-DIMENSIONAL FINITE-STRAIN VISCOELASTIC
//              DAMAGE MODEL: FORMULATION AND COMPUTATIONAL ASPECTS", by
//              J.C.Simo, Computer Methods in Applied Mechanics and
//              Engineering 60 (1987) 153-173.
//              "COMPUTATIONAL INELASTICITY" by J.C.Simo & T.J.R.Hughes,
//              Springer, 1997.
//    Features:
//      Usage:


#include "ConstitutiveModelFactory.h"
#include "ViscoElasticDamage.h"
#include <Core/Exceptions/InternalError.h>
#include <Core/Malloc/Allocator.h>
#include <fstream>
#include <iostream>
using namespace std;
using namespace Uintah;

ViscoElasticDamage::ViscoElasticDamage(ProblemSpecP& ps)
{
  // Constructor
  // Initialize deformationGradient
  ps->require("bulk_modulus",d_Bulk);
  ps->require("shear_modulus",d_Shear);
  ps->require("alpha",d_Alpha);
  ps->require("beta",d_Beta);
  ps->require("tau",d_Tau);
  ps->require("gamma",d_Gamma);
  ps->require("max_equiv_strain",maxEquivStrain);

  deformationGradient.Identity();
  bElBar.Identity();
  damageG = 1.0;        // No damage at beginning
  E_bar.set(0.0);
  // initialization PI function at time=0
  func_PI_n.set(0.0);
  // initialization Hbar function at time=0
  func_Hbar_n.Identity();
 
}

void ViscoElasticDamage::addParticleState(std::vector<const VarLabel*>& from,
                                   std::vector<const VarLabel*>& to)
{
   throw InternalError("ViscoElasticDamage won't work");
}

ViscoElasticDamage::ViscoElasticDamage(double bulk, double shear,
                                double alpha, double beta, 
                                double tau, double gamma, double strainmax): 
  d_Bulk(bulk),d_Shear(shear),d_Alpha(alpha),d_Beta(beta),
  d_Tau(tau),d_Gamma(gamma), maxEquivStrain(strainmax)
{
  // Main constructor
  // Initialize deformationGradient

  deformationGradient.Identity();
  bElBar.Identity();
  damageG = 1.0;        // No damage at beginning
  E_bar.set(0.0);
  // initialization PI function at time=0
  func_PI_n.set(0.0);
  // initialization Hbar function at time=0
  func_Hbar_n.Identity();

}

ViscoElasticDamage::ViscoElasticDamage(const ViscoElasticDamage &cm):
  deformationGradient(cm.deformationGradient),
  bElBar(cm.bElBar),
  E_bar(cm.E_bar),
  current_E_bar(cm.current_E_bar),
  stressTensor(cm.stressTensor),
  func_PI_n(cm.func_PI_n),
  func_PI_nn(cm.func_PI_nn),
  func_Hbar_n(cm.func_Hbar_n),
  func_Hbar_nn(cm.func_Hbar_nn),
  d_Bulk(cm.d_Bulk),
  d_Shear(cm.d_Shear),
  d_Alpha(cm.d_Alpha),
  d_Beta(cm.d_Beta),
  d_Tau(cm.d_Tau),
  d_Gamma(cm.d_Gamma),
  damageG(cm.damageG),
  maxEquivStrain(cm.maxEquivStrain)
 
{
  // Copy constructor
 
}

ViscoElasticDamage::~ViscoElasticDamage()
{
  // Destructor
 
}

void ViscoElasticDamage::setBulk(double bulk)
{
  // Assign ViscoElasticDamage Bulk Modulus

  d_Bulk = bulk;
}

void ViscoElasticDamage::setShear(double shear)
{
  // Assign ViscoElasticDamage Shear Modulus

  d_Shear = shear;
}

void ViscoElasticDamage::setDamageParameters(double alpha, double beta)
{
  // Assign ViscoElasticDamage Damage Parameters

  d_Alpha = alpha;
  d_Beta = beta;
}

void ViscoElasticDamage::setViscoelasticParameters(double tau, double gamma)
{
  // Assign Viscoelastic Parameters

  d_Tau = tau;
  d_Gamma = gamma;
}

void ViscoElasticDamage::setMaxEquivStrain(double strainmax)
{
  // Assign Viscoelastic Parameters

  maxEquivStrain = strainmax;
}

void ViscoElasticDamage::setStressTensor(Matrix3 st)
{
  // Assign the stress tensor (3 x 3 Matrix)

  stressTensor = st;

}

void ViscoElasticDamage::setDeformationMeasure(Matrix3 dg) 
{
  // Assign the deformation gradient tensor (3 x 3 Matrix)

  deformationGradient = dg;

}

Matrix3 ViscoElasticDamage::getStressTensor() const
{
  // Return the stress tensor (3 x 3 Matrix)

  return stressTensor;

}

Matrix3 ViscoElasticDamage::getDeformationMeasure() const
{
  // Return the strain tensor (3 x 3 Matrix)

  return deformationGradient;

}

std::vector<double> ViscoElasticDamage::getMechProps() const
{
  // Return bulk and shear modulus

  std::vector<double> props(8,0.0);

  props[0] = d_Bulk;
  props[1] = d_Shear;
  props[2] = d_Alpha;
  props[3] = d_Beta;
  props[4] = d_Tau;
  props[5] = d_Gamma;
  props[6] = maxEquivStrain;
  props[7] = damageG;

  return props;

}

void ViscoElasticDamage::computeStressTensor(const PatchSubset* patches,
                                             const MPMMaterial* matl,
                                             DataWarehouse* old_dw,
                                             DataWarehouse* new_dw)
{
#ifdef WONT_COMPILE_YET
  Matrix3 bElBarTrial,shearTrial,fbar,F_bar,C_bar,C_nn;
  double J,p;
  double equiv_strain;
  double dt_tau;
  Matrix3 damage_normal,Ebar_increament,dev_Snn,temp_par;
  Matrix3 deformationGradientInc;
  double onethird = (1.0/3.0);
  double damage_flag = 0.0;
  Matrix3 Identity;

  Identity.Identity();

  // Calculate the stress Tensor (symmetric 3 x 3 Matrix) given the
  // time step and the velocity gradient and the material constants

  
  // Compute the deformation gradient increment using the time_step
  // velocity gradient
  // F_n^np1 = dudx * dt + Identity
  deformationGradientInc = velocityGradient * time_step + Identity;

  // Update the deformation gradient tensor to its time n+1 value.
  deformationGradient = deformationGradientInc * deformationGradient;

  fbar = deformationGradientInc *
                        pow(deformationGradientInc.Determinant(),-onethird);

  F_bar = deformationGradient*pow(deformationGradient.Determinant(),-onethird);

  C_bar = F_bar * F_bar.Transpose();
  C_nn = deformationGradient * deformationGradient.Transpose();

  // Deviatoric-Elastic part of the Lagrangian Strain Tensor
  current_E_bar = (C_bar - Identity) / 2.0;

  // Calculate equivalent strain
  equiv_strain = sqrt(d_Shear*(C_bar.Trace() - 3.0));
  maxEquivStrain = (maxEquivStrain > equiv_strain) ? 
                                maxEquivStrain : equiv_strain;

  // Normal of the damage surface
  damage_normal = Identity * d_Shear / (2.0*equiv_strain);

  // Check damage criterion
  if ( equiv_strain == maxEquivStrain && maxEquivStrain != 0.0) 
  // on damage surface
  {
    Ebar_increament = current_E_bar - E_bar;

    for( int damage_i=1; damage_i<=3; damage_i++)
       for( int damage_j=1; damage_j<=3; damage_j++)
          damage_flag += damage_normal(damage_i, damage_j) *
                        Ebar_increament(damage_i, damage_j);

       if (damage_flag > 0.0)   // loading: further damage 
          damageG = d_Beta + (1-d_Beta) * (1-exp(-maxEquivStrain/d_Alpha))
                        / (maxEquivStrain/d_Alpha);
  } // end if
  E_bar = current_E_bar;

  Matrix3 temp_matrix3;
  temp_matrix3 = Identity*d_Shear/2.0 - C_nn.Inverse()*
                                        C_nn.Trace()*d_Shear/6.0;
  func_PI_nn =  temp_matrix3 * damageG;
  dt_tau = time_step/d_Tau;
  func_Hbar_nn = func_Hbar_n * exp(-dt_tau) + 
                 (func_PI_nn - func_PI_n) * (1.0-exp(-dt_tau))/dt_tau;
  func_PI_n = func_PI_nn;
  func_Hbar_n = func_Hbar_nn;
  
  temp_par = func_PI_nn*d_Gamma + func_Hbar_nn*(1.0 - d_Gamma);
  double temp_double = 0.0; 
  for( int damage_i=1; damage_i<=3; damage_i++)
       for( int damage_j=1; damage_j<=3; damage_j++)
          temp_double += temp_par(damage_i, damage_j) * 
                                        C_nn(damage_i, damage_j); 

  dev_Snn = (temp_par - C_nn.Inverse() * temp_double/3.0) *
                pow(deformationGradient.Determinant(),-2.0/3.0);

  shearTrial = deformationGradient*dev_Snn*deformationGradient.Transpose();

  // get the volumetric part of the deformation
  J = deformationGradient.Determinant();

  // get the hydrostatic part of the stress
  p = 0.5*d_Bulk*(J - 1.0/J);

  // compute the total stress (volumetric + deviatoric)

  stressTensor = Identity*J*p + shearTrial;

  bElBar = bElBarTrial;
#endif
}

double ViscoElasticDamage::computeStrainEnergy(const Patch* patch,
                                               const MPMMaterial* matl,
                                               DataWarehouse* new_dw)
{
#ifdef WONT_COMPILE_YET

  double strainenergy = 1;

  return strainenergy;
#endif

}

void ViscoElasticDamage::initializeCMData(const Patch* patch,
                      const MPMMaterial* matl,
                      DataWarehouse* new_dw)
{
}

double ViscoElasticDamage::getLambda() const
{
  // Return the Lame constant lambda

  double lambda;

  lambda = d_Bulk - .6666666667*d_Shear;

  return lambda;

}

double ViscoElasticDamage::getMu() const
{
  // Return the Lame constant mu

  return d_Shear;

}

void ViscoElasticDamage::readParameters(ProblemSpecP ps, double *p_array)
{
  
  ps->require("bulk_modulus",p_array[0]);
  ps->require("shear_modulus",p_array[1]);
  ps->require("alpha",p_array[2]);
  ps->require("beta",p_array[3]);
  ps->require("tau",p_array[4]);
  ps->require("gamma",p_array[5]);
  ps->require("max_equiv_strain",p_array[6]);
  
}

void ViscoElasticDamage::writeParameters(ofstream& out, double *p_array)
{
  out << p_array[0] << " " << p_array[1] << " ";
  out << p_array[2] << " " << p_array[3] << " ";
  out << p_array[4] << " " << p_array[5] << " ";
  out << p_array[6] << " "; 
}

ConstitutiveModel* ViscoElasticDamage::readParametersAndCreate(ProblemSpecP ps)
{
  double p_array[7];
  readParameters(ps, p_array);
  return(create(p_array));
}

void ViscoElasticDamage::writeRestartParameters(ofstream& out) const
{
  out << getType() << " ";
  out << d_Bulk << " " << d_Shear << " ";
  out << d_Alpha << " " << d_Beta << " ";
  out << d_Tau << " " << d_Gamma << " ";
  out << maxEquivStrain << " ";
  out << (getDeformationMeasure())(1,1) << " "
      << (getDeformationMeasure())(1,2) << " "
      << (getDeformationMeasure())(1,3) << " "
      << (getDeformationMeasure())(2,1) << " "
      << (getDeformationMeasure())(2,2) << " "
      << (getDeformationMeasure())(2,3) << " "
      << (getDeformationMeasure())(3,1) << " "
      << (getDeformationMeasure())(3,2) << " "
      << (getDeformationMeasure())(3,3) << endl;
}

ConstitutiveModel* ViscoElasticDamage::readRestartParametersAndCreate(ProblemSpecP ps)
{
#if 0
  Matrix3 dg(0.0);
  ConstitutiveModel *cm = readParametersAndCreate(in);
  
  in >> dg(1,1) >> dg(1,2) >> dg(1,3)
     >> dg(2,1) >> dg(2,2) >> dg(2,3)
     >> dg(3,1) >> dg(3,2) >> dg(3,3);
  cm->setDeformationMeasure(dg);
  
  return(cm);
#endif
}

ConstitutiveModel* ViscoElasticDamage::create(double *p_array)
{
  return(scinew ViscoElasticDamage(p_array[0],p_array[1],p_array[2],p_array[3],
                p_array[4],p_array[5],p_array[6]));
}

int ViscoElasticDamage::getType() const
{
  //  return(ConstitutiveModelFactory::CM_VISCOELASTIC_DAMAGE);
}

std::string ViscoElasticDamage::getName() const
{
  return("Viscoelastic-Damage");
}

int ViscoElasticDamage::getNumParameters() const
{
  return(7);
}

void ViscoElasticDamage::printParameterNames(ofstream& out) const
{
  out << "bulk" << endl
      << "shear" << endl
      << "d_Alpha" << endl
      << "d_Beta" << endl
      << "d_Tau" << endl
      << "d_Gamma" << endl
      << "maxEquivStrain" << endl;
}

void ViscoElasticDamage::addComputesAndRequires(Task* task,
                                             const MPMMaterial* matl,
                                             const PatchSet* patches) const

{
   cerr << "ViscoElasticDamage::addComputesAndRequires needs to be filled in\n";
}

ConstitutiveModel* ViscoElasticDamage::copy() const
{
  return( scinew ViscoElasticDamage(*this) );
}

int ViscoElasticDamage::getSize() const
{
  int s = 0;
  s += sizeof(double) * 7;  // properties
  s += sizeof(double) * 9;  // deformation gradient elements
  s += sizeof(double) * 6;  // stress tensor elements
  s += sizeof(double) * 9;  // bElBar elements
  s += sizeof(double) * 9;  // E_bar elements
  s += sizeof(double) * 9;  // current_E_bar elements
  s += sizeof(double) * 9;  // func_PI_n elements
  s += sizeof(double) * 9;  // func_PI_nn elements
  s += sizeof(double) * 9;  // func_Hbar_n elements
  s += sizeof(double) * 9;  // func_Hbar_nn elements
  s += sizeof(double) * 1;  // damG
  s += sizeof(int) * 1;     // type
  return(s);
}


