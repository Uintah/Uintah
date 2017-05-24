/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
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


#include <CCA/Components/MPM/ConstitutiveModel/PlasticityModels/BrittleDamage.h>

using namespace Uintah;
using std::cout;
using std::endl;
static DebugStream dbg("DamageModel", false);
//______________________________________________________________________
//      TODO:  
//

BrittleDamage::BrittleDamage( ProblemSpecP& dam_ps )
{
  Algorithm = DamageAlgo::brittle;
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
  ProblemSpecP matl_ps = dam_ps->getParent();
  ProblemSpecP cm_ps   = matl_ps->findBlock("constitutive_model");
  
  cm_ps->require("bulk_modulus",         d_brittle_damage.Bulk);
  cm_ps->require("shear_modulus",        d_brittle_damage.tauDev);

  //__________________________________
  //  Create labels
  const TypeDescription* P_dbl = ParticleVariable<double>::getTypeDescription();
  
  pFailureStressOrStrainLabel = VarLabel::create("p.epsf",          P_dbl );
  pFailureStressOrStrainLabel_preReloc = VarLabel::create("p.epsf+",P_dbl );
    
  pDamageLabel             = VarLabel::create("p.damage",     P_dbl );
  pDamageLabel_preReloc    = VarLabel::create("p.damage+",    P_dbl );
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
  VarLabel::destroy( pFailureStressOrStrainLabel );
  VarLabel::destroy( pFailureStressOrStrainLabel_preReloc );

  VarLabel::destroy( pDamageLabel );
  VarLabel::destroy( pDamageLabel_preReloc );
}
//______________________________________________________________________
//
void BrittleDamage::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP dam_ps = ps->appendChild("damage_model");
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
void
BrittleDamage::carryForward(const PatchSubset* patches,
                            const MPMMaterial* matl,
                            DataWarehouse*     old_dw,
                            DataWarehouse*     new_dw)
{
  const MaterialSubset* matls = matl->thisMaterial();
  bool replaceVar = true;
  new_dw->transferFrom( old_dw, pFailureStressOrStrainLabel,          patches, matls, replaceVar );
  new_dw->transferFrom( old_dw, pFailureStressOrStrainLabel_preReloc, patches, matls, replaceVar );

  new_dw->transferFrom( old_dw, pDamageLabel,                       patches, matls, replaceVar );
  new_dw->transferFrom( old_dw, pDamageLabel_preReloc,              patches, matls, replaceVar );
  
  new_dw->transferFrom( old_dw, d_lb->pLocalizedMPMLabel,           patches, matls, replaceVar );
  new_dw->transferFrom( old_dw, d_lb->pLocalizedMPMLabel_preReloc,  patches, matls, replaceVar );
}

//______________________________________________________________________
//
void
BrittleDamage::addParticleState(std::vector<const VarLabel*>& from,
                                std::vector<const VarLabel*>& to)
{
  from.push_back( pFailureStressOrStrainLabel );
  from.push_back( pDamageLabel );
  
  to.push_back( pFailureStressOrStrainLabel_preReloc );
  to.push_back( pDamageLabel_preReloc );
}

//______________________________________________________________________
//
void
BrittleDamage::addInitialComputesAndRequires(Task* task,
                                             const MPMMaterial* matl )
{
  int dwi = matl->getDWIndex();
  std::ostringstream mesg;
  mesg << "BrittleDamage::addInitialComputesAndRequires (matl:" << dwi <<  ")";
  printTask( dbg, mesg.str() );
  
  const MaterialSubset* matlset = matl->thisMaterial();
  task->computes( pFailureStressOrStrainLabel, matlset );
  task->computes( pDamageLabel,                matlset );
}
//______________________________________________________________________
//
void
BrittleDamage::initializeLabels(const Patch*       patch,
                                const MPMMaterial* matl,
                                DataWarehouse*     new_dw)
{
  int dwi = matl->getDWIndex();
  std::ostringstream mesg;
  mesg << "BrittleDamage::initializeLabels (matl:" << dwi << ")"; 
  printTask( patch, dbg, mesg.str() );
  
  ParticleSubset* pset = new_dw->getParticleSubset(matl->getDWIndex(), patch);

  ParticleVariable<double>   pFailureStrain;
  ParticleVariable<double>   pDamage;
  new_dw->allocateAndPut(pFailureStrain, pFailureStressOrStrainLabel, pset);
  new_dw->allocateAndPut(pDamage,        pDamageLabel,                pset);

  ParticleSubset::iterator iter = pset->begin();
  for(;iter != pset->end();iter++){
    pFailureStrain[*iter] = d_brittle_damage.r0b;
    pDamage[*iter]        = 0.0;
  }
}
//______________________________________________________________________
//
void
BrittleDamage::addComputesAndRequires(Task* task,
                                      const MPMMaterial* matl)
{
  printTask( dbg, "    BrittleDamage::addComputesAndRequires" );
  Ghost::GhostType  gnone = Ghost::None;
  const MaterialSubset* matls = matl->thisMaterial();

  VarLabel* pStressLabel      = VarLabel::find( "p.stress+" );
  VarLabel* pDefGradLabel     = VarLabel::find( "p.deformationMeasure+" );
  VarLabel* pParticleIDLabel  = VarLabel::find( "p.particleID" );  
  VarLabel* pVolumeLabel      = VarLabel::find( "p.volume+" );
  VarLabel* TotalLocalizedParticleLabel  = VarLabel::find( "TotalLocalizedParticle" );


  task->requires( Task::OldDW, pFailureStressOrStrainLabel, matls, gnone);   
  task->requires( Task::OldDW, pParticleIDLabel,            matls, gnone);   
  task->requires( Task::OldDW, d_lb->pLocalizedMPMLabel,    matls, gnone);   
  task->requires( Task::NewDW, pDefGradLabel,               matls, gnone);   
  task->requires( Task::NewDW, pVolumeLabel,                matls, gnone);
  task->requires( Task::OldDW, pDamageLabel,                matls, gnone);   
    
  task->modifies( pStressLabel,                         matls );
  task->computes( pFailureStressOrStrainLabel_preReloc, matls );
  task->computes( d_lb->pLocalizedMPMLabel_preReloc,    matls );
  task->computes( pDamageLabel_preReloc,                matls );
//  task->computes( TotalLocalizedParticleLabel );
}
//______________________________________________________________________
//
void
BrittleDamage::computeSomething( ParticleSubset    * pset,
                                 const MPMMaterial *,
                                 const Patch       * patch,
                                 DataWarehouse     * old_dw,
                                 DataWarehouse     * new_dw )
{
  printTask( patch, dbg, "    BrittleDamage::computeSomething" );

  constParticleVariable<int>     pLocalized;
  constParticleVariable<double>  pFailureStrain;
  constParticleVariable<long64>  pParticleID;
  constParticleVariable<Matrix3> pDefGrad;
  constParticleVariable<double>  pDamage;
  constParticleVariable<double>  pVolume;
  
  ParticleVariable<Matrix3>      pStress;
  ParticleVariable<int>          pLocalized_new;
  ParticleVariable<double>       pFailureStrain_new;
  ParticleVariable<double>       pDamage_new;

  VarLabel* pStressLabel      = VarLabel::find( "p.stress+" );
  VarLabel* pDefGradLabel     = VarLabel::find( "p.deformationMeasure+" );
  VarLabel* pParticleIDLabel  = VarLabel::find( "p.particleID" );
  VarLabel* pVolumeLabel      = VarLabel::find( "p.volume+" );

  old_dw->get(pLocalized,          d_lb->pLocalizedMPMLabel,    pset);     
  old_dw->get(pFailureStrain,      pFailureStressOrStrainLabel, pset);     
  old_dw->get(pParticleID,         pParticleIDLabel,            pset);     
  old_dw->get(pDamage,             pDamageLabel,                pset); 
  new_dw->get(pDefGrad,            pDefGradLabel,               pset);
  new_dw->get(pVolume,             pVolumeLabel,                pset);    

  new_dw->getModifiable(pStress,   pStressLabel,                pset);     

  new_dw->allocateAndPut(pLocalized_new,
                         d_lb->pLocalizedMPMLabel_preReloc,       pset);

  new_dw->allocateAndPut(pFailureStrain_new,
                         pFailureStressOrStrainLabel_preReloc,  pset);

  new_dw->allocateAndPut(pDamage_new,
                         pDamageLabel_preReloc,                 pset);

  BrittleDamageData bd = d_brittle_damage;  // for readabilty

    // Copy to new dw
  pFailureStrain_new.copyData( pFailureStrain );
  pLocalized_new.copyData( pLocalized );

  //__________________________________
  //
  ParticleSubset::iterator iter = pset->begin();
  for(; iter != pset->end(); iter++){
    particleIndex idx = *iter;
    
    pDamage_new[idx] = pDamage[idx];

    Matrix3 Identity, zero(0.0); Identity.Identity();
    double tau_b;  // current 'energy'

    // mean stress
    double pressure = (1.0/3.0)*pStress[idx].Trace();

    if (pressure <0.0) {

      //no damage if compressive
      if (pDamage[idx] <=0.0) { // previously no damage, do nothing
        return;
      }
      else {
        //previously damaged, deactivate damage?
        if ( bd.allowRecovery ) {  //recovery
          pStress[idx] = pStress[idx] * bd.recoveryCoeff;
          pDamage_new[idx]  = -pDamage[idx];         //flag damage to be negative
        }

        if ( bd.printDamage  ){
          cout << "Particle " << pParticleID[idx] << " damage halted: damage=" << pDamage_new[idx] << endl;
        }
        else {
          pStress[idx] = pStress[idx]*(1.0 - pDamage[idx]); // no recovery (default)
        }
      }
    } //end pDamage <=0.0

    //__________________________________
    // pressure >0.0; possible damage
    else {

      // Compute Finger tensor (left Cauchy-Green)
      Matrix3 bb = pDefGrad[idx] * pDefGrad[idx].Transpose();

      // Compute Eulerian strain tensor
      Matrix3 ee = (Identity - bb.Inverse())*0.5;

      // Compute the maximum principal strain
      double epsMax=0., epsMed=0., epsMin=0.;
      ee.getEigenValues( epsMax,epsMed,epsMin );

      // Young's modulus
      double young = 9.0 * bd.Bulk * bd.tauDev/\
                    (3.0 * bd.Bulk + bd.tauDev);

      tau_b = sqrt( young * epsMax * epsMax );

      //__________________________________
      //
      if ( tau_b > pFailureStrain[idx] ) {
        // further damage equivalent dimension of the particle
        double particleSize = pow( pVolume[idx], 1.0/3.0 );
        double r0b     = bd.r0b;
        double const_D = bd.constant_D;
        double const_C = r0b * particleSize * (1.0 + const_D ) \
                        /(bd.Gf * const_D) * log(1.0 + const_D);

        double d1      = 1.0 + const_D * exp( -const_C * ( tau_b - r0b ));
        double damage  = 0.999/const_D * ( (1.0 + const_D)/d1 - 1.0);

        // Restrict the maximum damage in a time step for stability reason.
        if ( (damage - pDamage[idx]) > bd.maxDamageInc ) {
          damage = pDamage[idx] + bd.maxDamageInc;
        }
        // Update threshold and damage
        pFailureStrain_new[idx] = tau_b;
        pDamage_new[idx] = damage;

        // Update stress
        pStress[idx] = pStress[idx] * ( 1.0 - damage );

        if ( bd.printDamage ){
          cout << "Particle " << pParticleID[idx] << " damaged: "
               << " damage=" << pDamage_new[idx] << " epsMax=" << epsMax
               << " tau_b=" << tau_b << endl;
        }
      }
      //__________________________________
      else {
        if ( pDamage[idx] == 0.0 ){
          return; // never damaged
        }

        //current energy less than previous; deactivate damage?
        if ( bd.allowRecovery ) { //recovery

          pStress[idx]     = pStress[idx] * bd.recoveryCoeff;
          pDamage_new[idx] = -pDamage[idx]; //flag it to be negative

          if ( bd.printDamage ){
            cout << "Particle " << pParticleID[idx] << " damage halted: damage="
                 << pDamage_new[idx] << endl;
          }
        }
        else { //no recovery (default)
          pStress[idx] = pStress[idx] * ( 1.0 - pDamage[idx] );

          if ( bd.printDamage ){
            cout << "Particle " << pParticleID[idx] << " damaged: "
                 << " damage=" << pDamage_new[idx] << " epsMax=" << epsMax
                 << " tau_b=" << tau_b << endl;
          }
        }
      } // end if tau_b > pFailureStrain
    } //end if pressure

  } // particle loop
}
