/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
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


#include <CCA/Components/MPM/Materials/ConstitutiveModel/PlasticityModels/JohnsonCookDamage.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Math/Gaussian.h>
#include <cmath>

using namespace Uintah;
static DebugStream dbg("DamageModel", false);
//______________________________________________________________________
//
JohnsonCookDamage::JohnsonCookDamage(ProblemSpecP& ps)
{
  Algorithm = DamageAlgo::johnson_cook;
  d_initialData.D0 = 0.0;
  d_initialData.Dc = 1.0e-10;
  
  d_initialData.D0_std  = 0.0; // Initial STD scalar damage
  d_initialData.dist    = "constant";

  ps->get( "initial_mean_scalar_damage",    d_initialData.D0);
  ps->get( "initial_std_scalar_damage",     d_initialData.D0_std);
  ps->get( "critical_scalar_damage",        d_initialData.Dc);
  ps->get( "initial_scalar_damage_distrib", d_initialData.dist);

  ps->require( "D1",d_initialData.D1);
  ps->require( "D2",d_initialData.D2);
  ps->require( "D3",d_initialData.D3);
  ps->require( "D4",d_initialData.D4);
  ps->require( "D5",d_initialData.D5);  
  
  const TypeDescription* P_dbl = ParticleVariable<double>::getTypeDescription();  
  pDamageLabel = VarLabel::create("p.damage",           P_dbl );
  pDamageLabel_preReloc = VarLabel::create("p.damage+", P_dbl );
  pPlasticStrainRateLabel_preReloc = VarLabel::find("p.plasticStrainRate+");
} 
//______________________________________________________________________
//         
JohnsonCookDamage::JohnsonCookDamage(const JohnsonCookDamage* cm)
{
  d_initialData.D1 = cm->d_initialData.D1;
  d_initialData.D2 = cm->d_initialData.D2;
  d_initialData.D3 = cm->d_initialData.D3;
  d_initialData.D4 = cm->d_initialData.D4;
  d_initialData.D5 = cm->d_initialData.D5;
  d_initialData.D0 = cm->d_initialData.D0;
  d_initialData.Dc = cm->d_initialData.Dc;
} 
//______________________________________________________________________
//         
JohnsonCookDamage::~JohnsonCookDamage()
{
  VarLabel::destroy(pDamageLabel);
  VarLabel::destroy(pDamageLabel_preReloc);
}
//______________________________________________________________________
//
void JohnsonCookDamage::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP damage_ps = ps->appendChild("damage_model");
  damage_ps->setAttribute("type","johnson_cook");

  damage_ps->appendElement("D1",d_initialData.D1);
  damage_ps->appendElement("D2",d_initialData.D2);
  damage_ps->appendElement("D3",d_initialData.D3);
  damage_ps->appendElement("D4",d_initialData.D4);
  damage_ps->appendElement("D5",d_initialData.D5);
  damage_ps->appendElement( "initial_mean_scalar_damage",    d_initialData.D0);
  damage_ps->appendElement( "initial_std_scalar_damage",     d_initialData.D0_std);
  damage_ps->appendElement( "critical_scalar_damage",        d_initialData.Dc);
  damage_ps->appendElement( "initial_scalar_damage_distrib", d_initialData.dist);
}

//______________________________________________________________________
//
void
JohnsonCookDamage::addParticleState(std::vector<const VarLabel*>& from,
                                    std::vector<const VarLabel*>& to)
{
  from.push_back( pDamageLabel );
  to.push_back(   pDamageLabel_preReloc );
}
//______________________________________________________________________
//
void 
JohnsonCookDamage::addInitialComputesAndRequires(Task* task,
                                           const MPMMaterial* matl )
{
  int dwi = matl->getDWIndex();
  std::ostringstream mesg;
  mesg << "JohnsonCookDamage::addInitialComputesAndRequires (matl:" << dwi <<  ")";
  printTask( dbg, mesg.str() );
  
  const MaterialSubset* matls = matl->thisMaterial();
  task->computes(pDamageLabel, matls);
}
//______________________________________________________________________
//
void
JohnsonCookDamage::initializeLabels(const Patch       * patch,
                                    const MPMMaterial * matl,       
                                    DataWarehouse     * new_dw)     
{
  int dwi = matl->getDWIndex();
  std::ostringstream mesg;
  mesg << "JohnsonCookDamage::initializeLabels (matl:" << dwi << ")";
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
JohnsonCookDamage::addComputesAndRequires(Task* task,
                                          const MPMMaterial* matl)
{
  printTask( dbg, "    JohnsonCookDamage::addComputesAndRequires" );
  Ghost::GhostType  gnone = Ghost::None;
  const MaterialSubset* matls = matl->thisMaterial();

//  VarLabel* TotalLocalizedParticleLabel  = VarLabel::find( "TotalLocalizedParticle" );

  task->requires( Task::OldDW, pDamageLabel,                     matls, gnone);
  task->requires( Task::OldDW, d_lb->pTemperatureLabel,          matls, gnone);      
  task->requires( Task::OldDW, d_lb->pLocalizedMPMLabel,        matls, gnone);

  task->requires( Task::NewDW, d_lb->pStressLabel_preReloc,      matls, gnone);      
  task->requires( Task::NewDW, pPlasticStrainRateLabel_preReloc, matls, gnone);

  task->computes( pDamageLabel_preReloc, matls );

//  task->computes(TotalLocalizedParticleLabel);
}

//______________________________________________________________________
//
void
JohnsonCookDamage::computeSomething( ParticleSubset    * pset,
                                     const MPMMaterial * matl,
                                     const Patch       * patch,
                                     DataWarehouse     * old_dw,
                                     DataWarehouse     * new_dw )
{
  printTask( patch, dbg, "    JohnsonCookDamage::computeSomething" );

  constParticleVariable<Matrix3> pStress;
  constParticleVariable<double>  pPlasticStrainRate;
  constParticleVariable<double>  pDamage_old;
  constParticleVariable<double>  pTemperature;
  constParticleVariable<int>     pLocalized;

  ParticleVariable<double>       pDamage;
  ParticleVariable<int>          pLocalizedNew;
  
  old_dw->get( pDamage_old,         pDamageLabel,                      pset);
  old_dw->get( pTemperature,        d_lb->pTemperatureLabel,           pset);

  new_dw->get( pStress,             d_lb->pStressLabel_preReloc,       pset);   
  new_dw->get( pPlasticStrainRate,  pPlasticStrainRateLabel_preReloc,  pset);

  new_dw->allocateAndPut( pDamage,  pDamageLabel_preReloc,             pset);

    // Get the time increment (delT)
  delt_vartype delT;
  old_dw->get(delT, d_lb->delTLabel, patch->getLevel());
  
  Matrix3 I; 
  I.Identity();
  
  const double Tr = matl->getRoomTemperature();
  const double Tm = matl->getMeltTemperature();

  //__________________________________
  //
  ParticleSubset::iterator iter = pset->begin();
  for(; iter != pset->end(); iter++){
    particleIndex idx = *iter;
    
//    if ( pPlasticStrainRate[idx] == 0.0 ){ // no plastic deformation or damage
//      continue;                            // Jim: please double check this
//    }
    double epdot    = pPlasticStrainRate[idx];  
    double sigMean  = pStress[idx].Trace()/3.0;
    Matrix3 sig_dev = pStress[idx] - I*sigMean;
    double sigEquiv = sqrt( (sig_dev.NormSquared())*1.5 );

    double sigStar = 0.0;
    if (sigEquiv != 0){
      sigStar = sigMean/sigEquiv;
    }
    if (sigStar > 1.5){
      sigStar = 1.5;
    }
    if (sigStar < -1.5){
      sigStar = -1.5;
    }

    double stressPart = d_initialData.D1 + d_initialData.D2 * exp(d_initialData.D3 * sigStar);

    double strainRatePart = 1.0;

    if (epdot < 1.0) { 
      strainRatePart = pow((1.0 + epdot),d_initialData.D4);
    }else{
      strainRatePart = 1.0 + d_initialData.D4*log(epdot);
    }

    double Tstar    = (pTemperature[idx] - Tr)/(Tm - Tr);
    double tempPart = 1.0 + d_initialData.D5*Tstar;

    // Calculate the updated scalar damage parameter
    double epsFrac = stressPart*strainRatePart*tempPart;
    if (epsFrac < d_initialData.Dc){
      pDamage[idx] = pDamage_old[idx];
    }

    // Calculate plastic strain increment
    double epsInc = epdot*delT;
    pDamage[idx] = pDamage_old[idx] + epsInc/epsFrac;
    if (pDamage[idx] < d_initialData.Dc){
      pDamage[idx] = 0.0;
    }
  #if 0
  if( epdot != 0 && Tstar != 0){
  std::cout.precision(16);
  std::cout << "epdot: " << epdot 
            << " T: " << pTemperature[idx] 
            << " delT: " << delT 
            << " tolerance: " << d_initialData.Dc << std::endl;

  std::cout << "  pStress: " << pStress[idx] << std::endl;
  
  std::cout << "  sigstar = " << sigStar 
            << " epdotStar = " << epdot
            << " Tstar = " << Tstar << std::endl;
       
  std::cout << "  e_inc = " << epsInc
            << " e_f = " << epsFrac
            << " D_n = " << pDamage_old[idx] 
            << " D_n+1 = " << pDamage[idx] << std::endl;
  }
  #endif

  }  // pset loop
}
