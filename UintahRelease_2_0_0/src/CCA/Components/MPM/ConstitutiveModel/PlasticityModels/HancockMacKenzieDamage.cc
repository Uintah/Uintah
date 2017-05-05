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


#include <CCA/Components/MPM/ConstitutiveModel/PlasticityModels/HancockMacKenzieDamage.h>
#include <Core/Math/Gaussian.h>
#include <cmath>

using namespace Uintah;
static DebugStream dbg("DamageModel", false);
//______________________________________________________________________
//
HancockMacKenzieDamage::HancockMacKenzieDamage(ProblemSpecP& ps)
{
  Algorithm = DamageAlgo::hancock_mackenzie;
  // defaults
  d_initialData.D0      = 0.0; // Initial scalar damage
  d_initialData.D0_std  = 0.0; // Initial STD scalar damage
  d_initialData.Dc      = 1.0; // Critical scalar damage
  d_initialData.dist = "constant";

  ps->get("initial_mean_scalar_damage",        d_initialData.D0);
  ps->get("initial_std_scalar_damage",         d_initialData.D0_std);
  ps->get("critical_scalar_damage",            d_initialData.Dc);
  ps->get("initial_scalar_damage_distrib",     d_initialData.dist);

  const TypeDescription* P_dbl = ParticleVariable<double>::getTypeDescription();
  pDamageLabel = VarLabel::create("p.damage",           P_dbl );
  pDamageLabel_preReloc = VarLabel::create("p.damage+", P_dbl );
  
  pPlasticStrainRateLabel_preReloc = VarLabel::find("p.plasticStrainRate+");
}
//______________________________________________________________________
//
HancockMacKenzieDamage::HancockMacKenzieDamage(const HancockMacKenzieDamage* cm)
{
  d_initialData.D0  = cm->d_initialData.D0;
  d_initialData.Dc  = cm->d_initialData.Dc;
}
//______________________________________________________________________
//
HancockMacKenzieDamage::~HancockMacKenzieDamage()
{
  VarLabel::destroy(pDamageLabel);
  VarLabel::destroy(pDamageLabel_preReloc);
}
//______________________________________________________________________
//
void HancockMacKenzieDamage::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP dam_ps = ps->appendChild("damage_model");
  dam_ps->setAttribute("type","hancock_mackenzie");


  dam_ps->appendElement("initial_mean_scalar_damage",    d_initialData.D0);
  dam_ps->appendElement("initial_std_scalar_damage",     d_initialData.D0_std);
  dam_ps->appendElement("critical_scalar_damage",        d_initialData.Dc);
  dam_ps->appendElement("initial_scalar_damage_distrib", d_initialData.dist);
}
//______________________________________________________________________
//
void
HancockMacKenzieDamage::addParticleState(std::vector<const VarLabel*>& from,
                                         std::vector<const VarLabel*>& to)
{
  from.push_back( pDamageLabel );
  to.push_back(   pDamageLabel_preReloc );
}

//______________________________________________________________________
//
void
HancockMacKenzieDamage::addInitialComputesAndRequires(Task* task,
                                                      const MPMMaterial* matl )
{
  int dwi = matl->getDWIndex();
  std::ostringstream mesg;
  mesg << "HancockMacKenzieDamage::addInitialComputesAndRequires (matl:" << dwi <<  ")";
  printTask( dbg, mesg.str() );
  
  const MaterialSubset* matls = matl->thisMaterial();
  task->computes(pDamageLabel, matls);
}
//______________________________________________________________________
//
void
HancockMacKenzieDamage::initializeLabels(const Patch      * patch,
                                         const MPMMaterial* matl,
                                         DataWarehouse    * new_dw)
{
  int dwi = matl->getDWIndex();
  std::ostringstream mesg;
  mesg << "HancockMacKenzieDamage::initializeLabels (matl:" << dwi << ")";
  printTask( patch, dbg, mesg.str() );

  ParticleSubset* pset = new_dw->getParticleSubset(matl->getDWIndex(), patch);

  ParticleVariable<double> pDamage;
  new_dw->allocateAndPut(pDamage, pDamageLabel, pset);

  //__________________________________
  //
  ParticleSubset::iterator iter = pset->begin();
  for(;iter != pset->end();iter++){
    pDamage[*iter] = d_initialData.D0;
  }

  if (d_initialData.dist != "constant") {

    Gaussian gaussGen(d_initialData.D0, d_initialData.D0_std, 0, 1,DBL_MAX);
    ParticleSubset::iterator iter = pset->begin();
    for(;iter != pset->end();iter++){

      // Generate a Gaussian distributed random number given the mean
      // damage and the std.
      pDamage[*iter] = fabs(gaussGen.rand(1.0));
    }
  }
}

//______________________________________________________________________
//
void
HancockMacKenzieDamage::addComputesAndRequires(Task* task,
                                              const MPMMaterial* matl)
{
  printTask( dbg, "    HancockMacKenzieDamage::addComputesAndRequires" );
  Ghost::GhostType  gnone = Ghost::None;
  const MaterialSubset* matls = matl->thisMaterial();

//  VarLabel* TotalLocalizedParticleLabel  = VarLabel::find( "TotalLocalizedParticle" );

  task->requires( Task::OldDW, pDamageLabel,                           matls, gnone);
  task->requires( Task::NewDW, d_lb->pStressLabel_preReloc,            matls, gnone);
  task->requires( Task::NewDW, pPlasticStrainRateLabel_preReloc, matls, gnone);
  task->computes( pDamageLabel_preReloc, matls );
//  task->computes(TotalLocalizedParticleLabel);
}

//______________________________________________________________________
//
void
HancockMacKenzieDamage::computeSomething( ParticleSubset    * pset,
                                          const MPMMaterial * matl,
                                          const Patch       * patch,
                                          DataWarehouse     * old_dw,
                                          DataWarehouse     * new_dw )
{
  printTask( patch, dbg, "    HancockMacKenzieDamage::computeSomething" );

  constParticleVariable<Matrix3> pStress;
  constParticleVariable<double>  pPlasticStrainRate;
  constParticleVariable<double>  pDamage_old;
  ParticleVariable<double>       pDamage;
  
  old_dw->get( pDamage_old,            pDamageLabel,                      pset);
  new_dw->get( pStress,                d_lb->pStressLabel_preReloc,       pset);
  new_dw->get( pPlasticStrainRate,     pPlasticStrainRateLabel_preReloc,  pset);
  new_dw->allocateAndPut( pDamage,     pDamageLabel_preReloc,             pset);

    // Get the time increment (delT)
  delt_vartype delT;
  old_dw->get(delT, d_lb->delTLabel, patch->getLevel());

  //__________________________________
  //
  ParticleSubset::iterator iter = pset->begin();
  for(; iter != pset->end(); iter++){
    particleIndex idx = *iter;

      // Calculate plastic strain increment
    double epsInc = pPlasticStrainRate[idx]*delT;

    // Compute hydrostatic stress and equivalent stress
    double sig_h = pStress[idx].Trace()/3.0;
    Matrix3 I;
    I.Identity();
    Matrix3 sig_dev = pStress[idx] - I*sig_h;
    double sig_eq   = 1.0;
    if(sig_h>0.0){
     sig_eq = sqrt( (sig_dev.NormSquared())*1.5);
    }

    // Calculate the updated scalar damage parameter
    pDamage[idx] = pDamage_old[idx] + (1.0/1.65) * epsInc * exp( 1.5*sig_h/sig_eq );

  }  // pset loop
}
