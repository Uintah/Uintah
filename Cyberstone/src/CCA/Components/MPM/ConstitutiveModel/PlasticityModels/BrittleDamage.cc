/*
 * The MIT License
 *
 * Copyright (c) 1997-2016 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */


#include "BrittleDamage.h"

using namespace Uintah;
using std::cout;
using std::endl;
//______________________________________________________________________
//
BrittleDamage::BrittleDamage( ProblemSpecP& dam_ps)
{
  std::cout << "BrittleDamage constructor" << std::endl;
  d_brittle_damage.r0b          = 57.0;     // Initial energy threshold
  d_brittle_damage.Gf           = 11.2;     // Fracture energy
  d_brittle_damage.constant_D   = 0.1;      // Shape constant in softening function
  d_brittle_damage.maxDamageInc = 0.1;      // Maximum damage in a time step
  d_brittle_damage.allowRecovery= false;    // Allow recovery
  d_brittle_damage.recoveryCoeff= 1.0;      // Fraction of recovery if allowed
  d_brittle_damage.printDamage  = false;    // Print damage

  dam_ps->get("brittle_damage_initial_threshold",   d_brittle_damage.r0b);
  dam_ps->get("brittle_damage_fracture_energy",     d_brittle_damage.Gf);
  dam_ps->get("brittle_damage_constant_D",          d_brittle_damage.constant_D);
  dam_ps->get("brittle_damage_max_damage_increment",d_brittle_damage.maxDamageInc);
  dam_ps->get("brittle_damage_allowRecovery",       d_brittle_damage.allowRecovery);
  dam_ps->get("brittle_damage_recoveryCoeff",       d_brittle_damage.recoveryCoeff);
  dam_ps->get("brittle_damage_printDamage",         d_brittle_damage.printDamage);

  if (d_brittle_damage.recoveryCoeff <0.0 || d_brittle_damage.recoveryCoeff>1.0){
    std::cerr << "brittle_damage_recoveryCoeff must be between 0.0 and 1.0" << std::endl;
  }
  ProblemSpecP cm_ps = dam_ps->getParent();
  cm_ps->require("bulk_modulus",         d_brittle_damage.Bulk);
  cm_ps->require("shear_modulus",        d_brittle_damage.tauDev);
  
}
//______________________________________________________________________
//
BrittleDamage::BrittleDamage(const BrittleDamage* )
{
}
//______________________________________________________________________
//
BrittleDamage::~BrittleDamage()
{
}
//______________________________________________________________________
//
void BrittleDamage::outputProblemSpec(ProblemSpecP& cm_ps)
{
  ProblemSpecP dam_ps = cm_ps->appendChild("damage");
  dam_ps->setAttribute("type","Brittle");

  dam_ps->appendElement("brittle_damage_initial_threshold", d_brittle_damage.r0b);
  dam_ps->appendElement("brittle_damage_fracture_energy",   d_brittle_damage.Gf);
  dam_ps->appendElement("brittle_damage_constant_D",        d_brittle_damage.constant_D);
  dam_ps->appendElement("brittle_damage_max_damage_increment", d_brittle_damage.maxDamageInc);
  dam_ps->appendElement("brittle_damage_allowRecovery",        d_brittle_damage.allowRecovery);
  dam_ps->appendElement("brittle_damage_recoveryCoeff",        d_brittle_damage.recoveryCoeff);
  dam_ps->appendElement("brittle_damage_printDamage",          d_brittle_damage.printDamage);
}

//______________________________________________________________________
//
inline double
BrittleDamage::initialize()
{
  return 0.0;
}
//______________________________________________________________________
//
inline bool
BrittleDamage:: hasFailed(double )
{
  return false;
}
//______________________________________________________________________
//
double
BrittleDamage::computeScalarDamage(const double& ,
                                     const Matrix3& ,
                                     const double& ,
                                     const double& ,
                                     const MPMMaterial*,
                                     const double& ,
                                     const double& )
{
  return 0.0;
}
//______________________________________________________________________
//
 void 
 BrittleDamage::updateDamageAndModifyStress2(const Matrix3& defGrad,
                                             const double&  pFailureStrain,
                                             double&        pFailureStrain_new,
                                             const double&  pVolume,
                                             const double&  pDamage,
                                             double&        pDamage_new,
                                             Matrix3&       pStress,
                                             const long64   particleID)
 {
  Matrix3 Identity, zero(0.0); Identity.Identity();
  double tau_b;  // current 'energy'

  // mean stress
  double pressure = (1.0/3.0)*pStress.Trace();

  BrittleDamageData bd = d_brittle_damage;  // for readabilty
  //__________________________________
  // Check for damage (note that pFailureStrain is the energy threshold)
  pFailureStrain_new = pFailureStrain;

  if (pressure <0.0) {

    //no damage if compressive
    if (pDamage <=0.0) { // previously no damage, do nothing
      return;
    }
    else {
      //previously damaged, deactivate damage?
      if ( bd.allowRecovery ) {  //recovery
        pStress     = pStress * bd.recoveryCoeff;
        pDamage_new = -pDamage;         //flag damage to be negative
      }

      if ( bd.printDamage  ){
        cout << "Particle " << particleID << " damage halted: damage=" << pDamage_new << endl;
      }
      else {
        pStress = pStress*(1.0-pDamage); // no recovery (default)
      }
    }
  } //end pDamage <=0.0

  //__________________________________
  // pressure >0.0; possible damage
  else {

    // Compute Finger tensor (left Cauchy-Green)
    Matrix3 bb = defGrad * defGrad.Transpose();

    // Compute Eulerian strain tensor
    Matrix3 ee = (Identity - bb.Inverse())*0.5;

    // Compute the maximum principal strain
    double epsMax=0., epsMed=0., epsMin=0.;
    ee.getEigenValues( epsMax,epsMed,epsMin );

    // Young's modulus
    double young = 9.0 * d_brittle_damage.Bulk * d_brittle_damage.tauDev/\
                  (3.0 * d_brittle_damage.Bulk + d_brittle_damage.tauDev);

    tau_b = sqrt( young * epsMax * epsMax );

    //__________________________________
    //
    if ( tau_b > pFailureStrain ) {
      // further damage equivalent dimension of the particle
      double particleSize = pow(pVolume, 1.0/3.0);
      double r0b     = bd.r0b;
      double const_D = bd.constant_D;
      double const_C = r0b * particleSize * (1.0 + const_D ) \
                      /(bd.Gf * const_D) * log(1.0 + const_D);

      double d1    = 1.0 + const_D * exp( -const_C * ( tau_b - r0b ));
      double damage= 0.999/const_D * ( (1.0 + const_D)/d1 - 1.0);

      // Restrict the maximum damage in a time step for stability reason.
      if ( (damage - pDamage) > bd.maxDamageInc ) {
        damage = pDamage + bd.maxDamageInc;
      }
      // Update threshold and damage
      pFailureStrain_new = tau_b;
      pDamage_new = damage;

      // Update stress
      pStress = pStress * ( 1.0 - damage );

      if ( bd.printDamage ){
        cout << "Particle " << particleID << " damaged: "
             << " damage=" << pDamage_new << " epsMax=" << epsMax
             << " tau_b=" << tau_b << endl;
      }
    }
    //__________________________________
    else {
      if ( pDamage == 0.0 ){
        return; // never damaged
      }

      //current energy less than previous; deactivate damage?
      if ( bd.allowRecovery ) { //recovery

        pStress     = pStress * bd.recoveryCoeff;
        pDamage_new = -pDamage; //flag it to be negative

        if ( bd.printDamage ){
          cout << "Particle " << particleID << " damage halted: damage="
               << pDamage_new << endl;
        }
      }
      else { //no recovery (default)
        pStress = pStress * ( 1.0 - pDamage );

        if ( bd.printDamage ){
          cout << "Particle " << particleID << " damaged: "
               << " damage=" << pDamage_new << " epsMax=" << epsMax
               << " tau_b=" << tau_b << endl;
        }
      }
    } // end if tau_b > pFailureStrain
  } //end if pressure
}
