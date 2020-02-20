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
#include <CCA/Components/MPM/Materials/ConstitutiveModel/PlasticityModels/ErosionModel.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Math/Matrix3.h>
#include <string.h>

using namespace Uintah;
using std::cout;
using std::endl;
static DebugStream dbg("ErosionModel", false);


//______________________________________________________________________
//  This class and the methods within are created for each MPM material.
//______________________________________________________________________
//
ErosionModel::ErosionModel( ProblemSpecP   & matl_ps,
                            MPMFlags       * flag,
                            MaterialManager* materialManager)
{
  d_lb = scinew MPMLabel();
  d_materialManager = materialManager;
    
  printTask( dbg, "ErosionModel constructor (each Matl)" );
  //__________________________________
  //  Set erosion algorithm
  d_algo= erosionAlgo::none;
  std::string algo = "none";
  
  ProblemSpecP em_ps = matl_ps->findBlock("erosion");
  if( em_ps ) {
    em_ps->getAttribute("algorithm", algo);
    
    if (algo == "AllowNoTension"){
      d_algo      = erosionAlgo::AllowNoTension;
      d_doEorsion = true;
    }
    else if (algo == "ZeroStress"){
      d_algo      = erosionAlgo::ZeroStress;
      d_doEorsion = true;
    }
    else if (algo == "AllowNoShear"){
      d_algo      = erosionAlgo::AllowNoShear;
      d_doEorsion = true;
    }          
    em_ps->getWithDefault("char_time", d_charTime, 1.0e-99);        // Characteristic time for damage
//    cout << " //__________________________________d_Algo: " << d_algo << " algoName " << algo << " d_charTime: " << d_charTime << "  " <<( d_algo == erosionAlgo::none ) << endl;
  }
  
  d_algoName = algo;
  //__________________________________
  //  labels local to model
  const TypeDescription* P_dbl = ParticleVariable<double>::getTypeDescription();
  pTimeOfLocLabel          = VarLabel::create("p.timeofloc",   P_dbl );
  pTimeOfLocLabel_preReloc = VarLabel::create("p.timeofloc+",  P_dbl);  
}

//______________________________________________________________________
//
ErosionModel::~ErosionModel()
{
  printTask( dbg, "ErosionModel destructor" );
  delete d_lb;
  VarLabel::destroy( pTimeOfLocLabel );
  VarLabel::destroy( pTimeOfLocLabel_preReloc );
}

//______________________________________________________________________
//
void 
ErosionModel::outputProblemSpec( ProblemSpecP& ps )
{
  if(! d_doEorsion) {
    return;
  }  
  printTask( dbg, "ErosionModel::outputProblemSpec (each Matl)" );
  ProblemSpecP dam_ps = ps->appendChild("erosion");
  dam_ps->setAttribute("algorithm",   d_algoName);
  dam_ps->appendElement("char_time",  d_charTime );
}

//______________________________________________________________________
//
void
ErosionModel::carryForward(const PatchSubset* patches,
                           const MPMMaterial* matl,
                           DataWarehouse    * old_dw,
                           DataWarehouse    * new_dw)
{
  if(! d_doEorsion) {
    return;
  }
  //
  printTask( patches, dbg, "ErosionModel::carryForward" );
  const MaterialSubset* matls = matl->thisMaterial();
  bool replaceVar = true;
  new_dw->transferFrom( old_dw, pTimeOfLocLabel,          patches, matls, replaceVar );
  new_dw->transferFrom( old_dw, pTimeOfLocLabel_preReloc, patches, matls, replaceVar );
}

//______________________________________________________________________
//
void
ErosionModel::addParticleState(std::vector<const VarLabel*>& from,
                               std::vector<const VarLabel*>& to)
{
  if(! d_doEorsion) {
    return;
  }
  //
  printTask( dbg, "ErosionModel::addParticleState" );
  from.push_back( pTimeOfLocLabel );
  to.push_back(   pTimeOfLocLabel_preReloc );
}
//______________________________________________________________________
//
void 
ErosionModel::addInitialComputesAndRequires(Task* task,
                                            const MPMMaterial* matl )
{
  if(! d_doEorsion) {
    return;
  }
  //
  int dwi = matl->getDWIndex();
  std::ostringstream mesg;
  mesg << "ErosionModel::addInitialComputesAndRequires (matl:" << dwi <<  ")";
  printTask( dbg, mesg.str() );
  
  const MaterialSubset* matls = matl->thisMaterial();
  task->computes( pTimeOfLocLabel, matls );
}
//______________________________________________________________________
//
void 
ErosionModel::initializeLabels(const Patch       * patch,
                               const MPMMaterial * matl,
                               DataWarehouse     * new_dw)
{
  if(! d_doEorsion) {
    return;
  }

  //
  int dwi = matl->getDWIndex();
  std::ostringstream mesg;
  mesg << "ErosionModel::initializeLabels (matl:" << dwi << ")"; 
  printTask( patch, dbg, mesg.str() );
  
  ParticleSubset* pset = new_dw->getParticleSubset(dwi, patch);
  
  ParticleVariable<double> pTimeOfLoc;
  new_dw->allocateAndPut(pTimeOfLoc, pTimeOfLocLabel, pset);
  
  ParticleSubset::iterator iter;    

  for(iter = pset->begin();iter != pset->end();iter++){
    pTimeOfLoc[*iter] = -1.e99;
  }
}
//______________________________________________________________________
//
void
ErosionModel::addComputesAndRequires(Task* task,
                                     const MPMMaterial* matl)
{
  if(! d_doEorsion) {
    return;
  }
  //
  printTask( dbg, "    ErosionModel::addComputesAndRequires" );
  Ghost::GhostType  gnone = Ghost::None;
  const MaterialSubset* matls = matl->thisMaterial();
  
  task->requires(Task::OldDW, pTimeOfLocLabel, matls, gnone);
  task->modifies(d_lb->pStressLabel_preReloc,  matls); 
  task->computes(pTimeOfLocLabel_preReloc,     matls);
}

//______________________________________________________________________
//
void
ErosionModel::updateStress_Erosion( ParticleSubset * pset,
                                    DataWarehouse  * old_dw,
                                    DataWarehouse  * new_dw)
{
  if(! d_doEorsion) {
    return;
  }
  //
  constParticleVariable<double> pTimeOfLoc;
  constParticleVariable<int>    pLocalized_old;
  constParticleVariable<int>    pLocalized_new;
  
  ParticleVariable<Matrix3>     pStress;
  ParticleVariable<double>      pTimeOfLoc_new;
  
  old_dw->get( pTimeOfLoc,           pTimeOfLocLabel,               pset);
  old_dw->get( pLocalized_old, d_lb->pLocalizedMPMLabel,            pset);
  new_dw->get( pLocalized_new, d_lb->pLocalizedMPMLabel_preReloc,   pset);
  
  new_dw->getModifiable(  pStress,  d_lb->pStressLabel_preReloc,    pset);

  new_dw->allocateAndPut( pTimeOfLoc_new, pTimeOfLocLabel_preReloc, pset);
  pTimeOfLoc_new.copyData( pTimeOfLoc );

  // Get the current simulation time
  // double simTime = d_materialManager->getElapsedSimTime();
                         
  simTime_vartype simTime(0);
  old_dw->get( simTime, d_lb->simulationTimeLabel );

  //__________________________________
  // If the particle has failed, apply various erosion algorithms
  if ( d_algo != erosionAlgo::none ) {
  
    Matrix3 Identity, zero(0.0); 
    Identity.Identity();

    ParticleSubset::iterator iter = pset->begin();
    for(; iter != pset->end(); iter++){
      particleIndex idx = *iter;

      if (pLocalized_old[idx] != pLocalized_new[idx]) {
        pTimeOfLoc_new[idx] = simTime;
      }
      double failTime = simTime - pTimeOfLoc_new[idx];

      //__________________________________
      //  modify the stress
      if( pLocalized_new[idx] != 0 ) {     // THIS SHOULD BE pLocalized_new
        // Compute pressure
        double pressure = pStress[idx].Trace()/3.0;

        double D = exp(-failTime/d_charTime);

        //cout << "D = " << D << endl;

        switch (d_algo) {
        case erosionAlgo::AllowNoTension :
          if( pressure > 0.0 ){
            pStress[idx] *= D;
          }else{
            pStress[idx] = Identity*pressure;
          }
          break;
        case erosionAlgo::AllowNoShear:
          pStress[idx] = Identity*pressure;
          break;
          
        case erosionAlgo::ZeroStress:
          pStress[idx] *= D;
          break;
        default:
          InternalError("Illegal erosionAlgo in ErosionModel::updateStress_Erosion", __FILE__, __LINE__);
        }  // switch
      }  // is localized
    }  // pset loop 
  }  // errosion != none
}
//______________________________________________________________________
//
void 
ErosionModel::updateVariables_Erosion( ParticleSubset * pset,
                                       const ParticleVariable<int>    & pLocalized,
                                       const ParticleVariable<Matrix3>& pFOld,
                                       ParticleVariable<Matrix3>      & pFNew,
                                       ParticleVariable<Matrix3>      & pVelGrad )
{
  Matrix3 zero(0.0);
  Matrix3 Identity;
  Identity.Identity();

  if( d_algo == erosionAlgo::ZeroStress ){
    for( ParticleSubset::iterator iter = pset->begin(); iter != pset->end(); iter++ ){
      particleIndex idx = *iter;
    
      if( pLocalized[idx] ){
        pFNew[idx]    = pFOld[idx];
        pVelGrad[idx] = zero;
      }
    }
  }

  if( d_algo == erosionAlgo::AllowNoShear || d_algo == erosionAlgo::AllowNoTension ){
    double third = 1./3.;
    
    for(ParticleSubset::iterator iter = pset->begin(); iter != pset->end(); iter++){
      particleIndex idx = *iter;
      
      if( pLocalized[idx] ){
        double cbrtJ  = cbrt( pFNew[idx].Determinant() );
        pFNew[idx]    = cbrtJ * Identity;
        pVelGrad[idx] = third * pVelGrad[idx].Trace() * Identity;
      }
    }
  }
}
