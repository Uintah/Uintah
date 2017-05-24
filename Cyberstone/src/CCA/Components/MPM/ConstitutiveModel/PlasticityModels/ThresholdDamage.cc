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


#include <CCA/Components/MPM/ConstitutiveModel/PlasticityModels/ThresholdDamage.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Math/Gaussian.h>
#include <Core/Math/Weibull.h>

using namespace Uintah;
using std::cout;
using std::endl;
static DebugStream dbg("DamageModel", false);
//______________________________________________________________________
//
ThresholdDamage::ThresholdDamage( ProblemSpecP    & ps,
                                  MPMFlags        * flags,
                                  SimulationState * sharedState )
{
  Algorithm = DamageAlgo::threshold;
  printTask( dbg, "ThresholdDamage constructor" );

  d_epsf.mean     = 10.0;                 // Mean failure stress or strain
  d_epsf.std      = 0.0;                  // Std. Dev or Weibull mod. for failure stres or strain
  d_epsf.seed     = 0;                    // seed for weibull distribution generator
  d_epsf.dist     = "constant";
  d_epsf.scaling  = "none";
  // "exponent" is the value of n used in c=(Vbar/V)^(1/n)
  // By setting the default value to DBL_MAX, that makes 1/n=0, which makes c=1
  d_epsf.exponent = DBL_MAX;              // Exponent used in vol. scaling of failure criteria
  d_epsf.refVol   = 1.0;                  // Reference volume for scaling failure criteria

  ps->require("failure_criteria", d_failure_criteria);

  if(d_failure_criteria!="MaximumPrincipalStress" &&
     d_failure_criteria!="MaximumPrincipalStrain" &&
     d_failure_criteria!="MohrColoumb"){
     throw ProblemSetupException("<failure_criteria> must be either MaximumPrincipalStress, MaximumPrincipalStrain or MohrColoumb", __FILE__, __LINE__);
  }

  if( d_failure_criteria == "MohrColoumb" ){
    // The cohesion value that MC needs is the "mean" value in the
    // FailureStressOrStrainData struct
    ps->require("friction_angle", d_friction_angle);
    ps->require("tensile_cutoff_fraction_of_cohesion", d_tensile_cutoff);
  }

  ps->require("failure_mean",d_epsf.mean);        // Mean val. of failure stress/strain
  ps->get("failure_distrib", d_epsf.dist);        // "constant", "weibull" or "gauss"

  // Only require std if using a non-constant distribution
  if( d_epsf.dist != "constant" ){
    ps->require("failure_std", d_epsf.std);      //Std dev (Gauss) or Weibull modulus
  }

  ps->get("scaling", d_epsf.scaling);             // "none" or "kayenta"
  if( d_epsf.scaling != "none" ){
    // If doing some sort of scaling, require user to provide a reference volume
    ps->require("reference_volume",d_epsf.refVol);

    if( d_epsf.dist == "weibull" ){
      d_epsf.exponent=d_epsf.std;                 // By default, exponent is Weibull modulus, BUT
      ps->get("exponent", d_epsf.exponent);       // allow user to choose the exponent
   } else {
      // Force user to choose the exponent
      ps->require("exponent", d_epsf.exponent);
    }
  }
  ps->get("failure_seed",    d_epsf.seed);        // Seed for RN generator


  //__________________________________
  //  Create labels
  const TypeDescription* P_dbl = ParticleVariable<double>::getTypeDescription();
    
  pFailureStressOrStrainLabel = VarLabel::create("p.epsf",        P_dbl );
  pFailureStressOrStrainLabel_preReloc = VarLabel::create("p.epsf+",P_dbl );
}
//______________________________________________________________________
//
ThresholdDamage::ThresholdDamage(const ThresholdDamage* )
{
}
//______________________________________________________________________
//
ThresholdDamage::~ThresholdDamage()
{
  printTask( dbg, "ThresholdDamage destructor" );
  VarLabel::destroy( pFailureStressOrStrainLabel );
  VarLabel::destroy( pFailureStressOrStrainLabel_preReloc );
}
//______________________________________________________________________
//
void ThresholdDamage::outputProblemSpec(ProblemSpecP& ps)
{
  printTask( dbg, "ThresholdDamage::outputProblemSpec" );
  ProblemSpecP dam_ps = ps->appendChild("damage_model");
  dam_ps->setAttribute("type","Threshold");

  dam_ps->appendElement("failure_mean",     d_epsf.mean);
  dam_ps->appendElement("failure_std",      d_epsf.std);
  dam_ps->appendElement("failure_exponent", d_epsf.exponent);
  dam_ps->appendElement("failure_seed" ,    d_epsf.seed);
  dam_ps->appendElement("failure_distrib",  d_epsf.dist);
  dam_ps->appendElement("failure_criteria", d_failure_criteria);
  dam_ps->appendElement("scaling",          d_epsf.scaling);
  dam_ps->appendElement("exponent",         d_epsf.exponent);
  dam_ps->appendElement("reference_volume", d_epsf.refVol);

  if(d_failure_criteria=="MohrColoumb"){
    dam_ps->appendElement("friction_angle", d_friction_angle);
    dam_ps->appendElement("tensile_cutoff_fraction_of_cohesion",
                                           d_tensile_cutoff);
  }
}

//______________________________________________________________________
//
void
ThresholdDamage::carryForward(const PatchSubset* patches,
                              const MPMMaterial* matl,
                              DataWarehouse*     old_dw,
                              DataWarehouse*     new_dw)
{
  printTask( patches, dbg, "ThresholdDamage::carryForward" );
  const MaterialSubset* matls = matl->thisMaterial();
  bool replaceVar = true;
  new_dw->transferFrom( old_dw, pFailureStressOrStrainLabel,          patches, matls, replaceVar );
  new_dw->transferFrom( old_dw, pFailureStressOrStrainLabel_preReloc, patches, matls, replaceVar );

  new_dw->transferFrom( old_dw, d_lb->pLocalizedMPMLabel,             patches, matls, replaceVar );
  new_dw->transferFrom( old_dw, d_lb->pLocalizedMPMLabel_preReloc,    patches, matls, replaceVar );
}

//______________________________________________________________________
//
void
ThresholdDamage::addParticleState(std::vector<const VarLabel*>& from,
                                  std::vector<const VarLabel*>& to)
{
  printTask( dbg, "ThresholdDamage::addParticleState" );
  from.push_back( pFailureStressOrStrainLabel );
  to.push_back( pFailureStressOrStrainLabel_preReloc );
}

//______________________________________________________________________
//
void
ThresholdDamage::addInitialComputesAndRequires(Task* task,
                                               const MPMMaterial* matl)
{
  int dwi = matl->getDWIndex();
  std::ostringstream mesg;
  mesg << "ThresholdDamage::addInitialComputesAndRequires (matl:" << dwi <<  ")";
  printTask( dbg, mesg.str() );
  
  const MaterialSubset* matls = matl->thisMaterial();
  task->computes( pFailureStressOrStrainLabel, matls );
  
//  VarLabel* TotalLocalizedParticleLabel  = VarLabel::find( "TotalLocalizedParticle" );
//  task->computes(TotalLocalizedParticleLabel);
}

//______________________________________________________________________
//
void
ThresholdDamage::initializeLabels(const Patch       * patch,
                                  const MPMMaterial * matl,
                                  DataWarehouse     * new_dw)
{
  int dwi = matl->getDWIndex();
  std::ostringstream mesg;
  mesg << "ThresholdDamage::initializeLabels (matl:" << dwi << ")"; 
  printTask( patch, dbg, mesg.str() );
  
  ParticleSubset* pset = new_dw->getParticleSubset(dwi, patch);
  ParticleVariable<double>      pFailureStrain;
  constParticleVariable<double> pVolume;

  new_dw->get(pVolume,                   d_lb->pVolumeLabel,          pset);
  new_dw->allocateAndPut(pFailureStrain, pFailureStressOrStrainLabel, pset);
  
  ParticleSubset::iterator iter;
  //__________________________________
  //
  // Make the seed differ for each patch, otherwise each patch gets the
  // same set of random #s.
  int patchID      = patch->getID();
  int patch_div_32 = patchID/32;
  patchID          = patchID%32;
  unsigned int unique_seed = ((d_epsf.seed+patch_div_32+1) << patchID);

  if (d_epsf.dist == "gauss"){
    Gaussian gaussGen(d_epsf.mean,d_epsf.std, unique_seed,
                      d_epsf.refVol, d_epsf.exponent);

    for(iter = pset->begin();iter != pset->end();iter++){
      pFailureStrain[*iter] = fabs(gaussGen.rand(pVolume[*iter]));
    }
  } else if (d_epsf.dist == "weibull"){
    // Initialize a weibull random number generator
    Weibull weibGen(d_epsf.mean, d_epsf.std, d_epsf.refVol,
                    unique_seed, d_epsf.exponent);
    
    for(iter = pset->begin();iter != pset->end();iter++){
      pFailureStrain[*iter] = weibGen.rand(pVolume[*iter]);
    }
  } else if (d_epsf.dist == "uniform") {
    MusilRNG* randGen = scinew MusilRNG(unique_seed);

    for(iter = pset->begin();iter != pset->end();iter++){
      double rand     = (*randGen)();
      double range    = (2*rand - 1)*d_epsf.std;
      double cc       = pow(d_epsf.refVol/pVolume[*iter], 1.0/d_epsf.exponent);
      double fail_eps = cc*(d_epsf.mean + range);
      pFailureStrain[*iter] = fail_eps;
    }
    delete randGen;

  } else {
    for(iter = pset->begin();iter != pset->end();iter++){
      pFailureStrain[*iter] = d_epsf.mean;
    }
  }
}

//______________________________________________________________________
//
void
ThresholdDamage::addComputesAndRequires(Task* task,
                                        const MPMMaterial* matl)
{
  printTask( dbg, "    ThresholdDamage::addComputesAndRequires" );
  Ghost::GhostType  gnone = Ghost::None;
  const MaterialSubset* matls = matl->thisMaterial();

//  VarLabel* TotalLocalizedParticleLabel  = VarLabel::find( "TotalLocalizedParticle" );

  task->requires(Task::OldDW, pFailureStressOrStrainLabel,    matls, gnone);
  task->requires(Task::OldDW, d_lb->pParticleIDLabel,         matls, gnone);
  task->requires(Task::NewDW, d_lb->pDeformationMeasureLabel_preReloc,                  
                                                              matls, gnone);
  task->requires(Task::OldDW, d_lb->pLocalizedMPMLabel,       matls, gnone);
  
  task->modifies(d_lb->pStressLabel_preReloc,          matls);

  task->computes(pFailureStressOrStrainLabel_preReloc, matls);
  
  //pLocalizedMPM+ _can_ be computed upstream
  if( matl->is_pLocalizedPreComputed() ){
    task->modifies(d_lb->pLocalizedMPMLabel_preReloc,    matls);
  } else { 
    task->computes(d_lb->pLocalizedMPMLabel_preReloc,    matls);    
  }
       
  
//  task->computes(TotalLocalizedParticleLabel);
}
//______________________________________________________________________
//
void
ThresholdDamage::computeSomething( ParticleSubset    * pset,
                                   const MPMMaterial * matl,
                                   const Patch       * patch,
                                   DataWarehouse     * old_dw,
                                   DataWarehouse     * new_dw )
{
  printTask( patch, dbg, "    ThresholdDamage::computeSomething" );
  
  constParticleVariable<int>     pLocalized;
  constParticleVariable<double>  pFailureStrain;
  constParticleVariable<long64>  pParticleID;
  constParticleVariable<Matrix3> pDefGrad_new;
  
  ParticleVariable<Matrix3>      pStress;
  ParticleVariable<int>          pLocalized_new;
  ParticleVariable<double>       pFailureStrain_new;

  old_dw->get(pLocalized,               d_lb->pLocalizedMPMLabel,    pset);
  old_dw->get(pFailureStrain,           pFailureStressOrStrainLabel, pset);
  old_dw->get(pParticleID,              d_lb->pParticleIDLabel,      pset);
  new_dw->get(pDefGrad_new,             d_lb->pDeformationMeasureLabel_preReloc,           
                                                                     pset);
  new_dw->getModifiable(pStress,        d_lb->pStressLabel_preReloc, pset);


  new_dw->allocateAndPut(pFailureStrain_new,
                         pFailureStressOrStrainLabel_preReloc,  pset);

  //pLocalizedMPM+ _can_ be computed upstream
  if( matl->is_pLocalizedPreComputed() ){
    new_dw->getModifiable(pLocalized_new,
                         d_lb->pLocalizedMPMLabel_preReloc,     pset);    
  } else {
    new_dw->allocateAndPut(pLocalized_new,
                         d_lb->pLocalizedMPMLabel_preReloc,     pset);    
    for(ParticleSubset::iterator iter = pset->begin(); 
                                 iter != pset->end(); iter++){
      particleIndex idx = *iter;
        pLocalized_new[idx] = 0.0;
    }
  }

  Matrix3 defGrad(0.0);
  
  // Copy failure strains to new dw
  pFailureStrain_new.copyData(pFailureStrain);

  //__________________________________
  //
  ParticleSubset::iterator iter = pset->begin();
  for(; iter != pset->end(); iter++){
    particleIndex idx = *iter;

    defGrad = pDefGrad_new[idx];
    Matrix3 Identity, zero(0.0); Identity.Identity();

    // Find if the particle has failed
    if(pLocalized_new[idx]==0){
      pLocalized_new[idx] = pLocalized[idx];
    }

    if (pLocalized[idx] == 0 && pLocalized_new[idx] != -999){
      if(d_failure_criteria=="MaximumPrincipalStress"){

        double maxEigen=0., medEigen=0., minEigen=0.;
        pStress[idx].getEigenValues(maxEigen, medEigen, minEigen);

        //The first eigenvalue returned by "eigen" is always the largest
        if ( maxEigen > pFailureStrain[idx] ){
          pLocalized_new[idx] = 1;
        }
        if ( pLocalized[idx] != pLocalized_new[idx]) {
          cout << "Particle " << pParticleID[idx] << " has failed : MaxPrinStress = "
               << maxEigen << " eps_f = " << pFailureStrain[idx] << endl;
        }
      }
      else if( d_failure_criteria=="MaximumPrincipalStrain" ){
        // Compute Finger tensor (left Cauchy-Green)
        Matrix3 bb = defGrad * defGrad.Transpose();

        // Compute Eulerian strain tensor
        Matrix3 ee = (Identity - bb.Inverse())*0.5;

        double maxEigen=0., medEigen=0., minEigen=0.;
        ee.getEigenValues(maxEigen,medEigen,minEigen);

        if ( maxEigen > pFailureStrain[idx] ){
          pLocalized_new[idx] = 1;
        }
        if ( pLocalized[idx] != pLocalized_new[idx]) {
          cout << "Particle " << pParticleID[idx] << " has failed : eps = " << maxEigen
               << " eps_f = " << pFailureStrain[idx] << endl;
        }
      }
      else if( d_failure_criteria=="MohrColoumb" ){
        double maxEigen=0., medEigen=0., minEigen=0.;
        pStress[idx].getEigenValues(maxEigen, medEigen, minEigen);

        double cohesion = pFailureStrain[idx];

        double epsMax=0.;
        // Tensile failure criteria (max princ stress > d_tensile_cutoff*cohesion)
        if (maxEigen > d_tensile_cutoff * cohesion){
          pLocalized_new[idx] = 1;
          epsMax = maxEigen;
        }

        //  Shear failure criteria (max shear > cohesion + friction)
        double friction_angle = d_friction_angle*(M_PI/180.);

        if ( (maxEigen - minEigen)/2.0 > cohesion * cos(friction_angle)
             - (maxEigen + minEigen)*sin(friction_angle)/2.0){
          pLocalized_new[idx] = 2;
          epsMax = (maxEigen - minEigen)/2.0;
        }
        if (pLocalized[idx] != pLocalized_new[idx]) {
          cout << "Particle " << pParticleID[idx] << " has failed : maxPrinStress = "
               << epsMax << " cohesion = " << cohesion << endl;
        }
      } // Mohr-Coloumb
    } // pLocalized==0
  }  // pset loop
}
