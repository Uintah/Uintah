/*
 * The MIT License
 *
 * Copyright (c) 1997-2026 The University of Utah
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

#include <CCA/Components/ICE/Core/ICELabel.h>
#include <CCA/Components/ICE/CustomBCs/BoundaryCond.h>
#include <CCA/Components/ICE/Materials/ICEMaterial.h>
#include <CCA/Components/Models/SolidReactionModel/Ablation0.h>
#include <CCA/Components/MPM/Materials/MPMMaterial.h>
#include <CCA/Components/MPM/Core/MPMLabel.h>
#include <CCA/Ports/Output.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/DbgOutput.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Material.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Util/DOUT.hpp>


// Delete all of this probably...

#if 0
// Rate Models
#include <CCA/Components/Models/SolidReactionModel/NthOrderModel.h>
#include <CCA/Components/Models/SolidReactionModel/PowerModel.h>
#include <CCA/Components/Models/SolidReactionModel/ProutTompkinsModel.h>
#include <CCA/Components/Models/SolidReactionModel/DiffusionModel.h>
#include <CCA/Components/Models/SolidReactionModel/ContractingSphereModel.h>
#include <CCA/Components/Models/SolidReactionModel/ContractingCylinderModel.h>
#include <CCA/Components/Models/SolidReactionModel/AvaramiErofeevModel.h>

// Rate Constant Models
#include <CCA/Components/Models/SolidReactionModel/Arrhenius.h>
#include <CCA/Components/Models/SolidReactionModel/ModifiedArrhenius.h>
#endif

#include <iostream>

using namespace Uintah;
using namespace std;
//__________________________________
//  setenv SCI_DEBUG "MODELS_DOING_COUT:+"
//  MODELS_DOING_COUT:   dumps when tasks are scheduled and performed

Dout dout_models_ab0("Ablation0", "Models::Ablation0", "Prints task scheduling & execution", false);
//______________________________________________________________________
//
Ablation0::Ablation0(const ProcessorGroup* myworld,
                                       const MaterialManagerP& materialManager,
                                       const ProblemSpecP& params,
                                       const ProblemSpecP& prob_spec)
  : ModelInterface(myworld, materialManager),
    d_params(params), d_prob_spec(prob_spec)
{
  d_myMatls = 0;
  Ilb = scinew ICELabel();
  Mlb = scinew MPMLabel();
  d_saveConservedVars = scinew saveConservedVars();

  // Labels
  reactedFractionLabel   = VarLabel::create("F",
                                       CCVariable<double>::getTypeDescription());
  delFLabel              = VarLabel::create("delF",
                                       CCVariable<double>::getTypeDescription());

  totalMassBurnedLabel  = VarLabel::create( "totalMassBurned",
                                            sum_vartype::getTypeDescription() );
  totalHeatReleasedLabel= VarLabel::create( "totalHeatReleased",
                                            sum_vartype::getTypeDescription() );
}
//______________________________________________________________________
//
Ablation0::~Ablation0()
{
  DOUTR( dout_models_ab0, " Doing: Ablation0 destructor ");
#if 0
  delete d_rateConstantModel;
  delete d_rateModel;
#endif

  delete Ilb;
  delete Mlb;
  delete d_saveConservedVars;

  VarLabel::destroy(reactedFractionLabel);
  VarLabel::destroy(delFLabel);
  VarLabel::destroy(totalMassBurnedLabel);
  VarLabel::destroy(totalHeatReleasedLabel);

  if(d_myMatls && d_myMatls->removeReference()){
    delete d_myMatls;
  }
}
//______________________________________________________________________
//
void Ablation0::outputProblemSpec(ProblemSpecP& ps)
{
  DOUTR( dout_models_ab0, " Ablation0::outputProblemSpec ");

  ProblemSpecP model_ps = ps->appendChild("Model");
  model_ps->setAttribute("type","Ablation0");
  ProblemSpecP srm_ps = model_ps->appendChild( "Ablation0" );

  srm_ps->appendElement("fromMaterial",d_fromMaterial);
  srm_ps->appendElement("toMaterial",  d_doMaterial);
  srm_ps->appendElement("E0",   d_E0);

//  d_rateConstantModel->outputProblemSpec( srm_ps );
//  d_rateModel        ->outputProblemSpec( srm_ps );
}
//______________________________________________________________________
//
void Ablation0::problemSetup(GridP& grid,
                                      const bool isRestart)
{

  DOUTR( dout_models_ab0, " Ablation0::problemSetup ");

  ProblemSpecP srm_ps = d_params->findBlock("Ablation0");
  // Get base includes
  srm_ps->require("fromMaterial",d_fromMaterial);
  srm_ps->require("toMaterial",  d_doMaterial);
  d_E0 = 0.;
  srm_ps->require("E0",          d_E0);

#if 0
  ProblemSpecP rcm_ps = srm_ps->findBlock("RateConstantModel");
  ProblemSpecP rm_ps  = srm_ps->findBlock("RateModel");

  //__________________________________
  //  Bulletproofing
  if(!rcm_ps){
    throw ProblemSetupException("Ablation0: Cannot find RateConstantModel", __FILE__, __LINE__);
  }

  if(!rm_ps){
    throw ProblemSetupException("Ablation0: Cannot find RateModel", __FILE__, __LINE__);
  }

  //__________________________________
  // Create the rate constant model
  string modelType;
  if(!rcm_ps->getAttribute("type", modelType)){
    throw ProblemSetupException("Ablation0: Cannot find type for RateConstantModel", __FILE__, __LINE__);
  }
  if(modelType == "Arrhenius"){
    d_rateConstantModel = scinew Arrhenius(rcm_ps);
  }
  if(modelType == "ModifiedArrhenius"){
    d_rateConstantModel = scinew ModifiedArrhenius(rcm_ps);
  }

  //__________________________________
  // Create the rate model
  if(!rm_ps->getAttribute("type", modelType)){
    throw ProblemSetupException("Ablation0: Cannot find type for RateModel", __FILE__, __LINE__);
  }

  if(modelType == "AvaramiErofeev"){
    d_rateModel = scinew AvaramiErofeevModel(rm_ps);
  }
  if(modelType == "ContractingCylinder"){
    d_rateModel = scinew ContractingCylinderModel(rm_ps);
  }
  if(modelType == "ContractingSphere"){
    d_rateModel = scinew ContractingSphereModel(rm_ps);
  }
  if(modelType == "Diffusion"){
    d_rateModel = scinew DiffusionModel(rm_ps);
  }
  if(modelType == "Power"){
    d_rateModel = scinew PowerModel(rm_ps);
  }
  if(modelType == "ProutTompkins"){
    d_rateModel = scinew ProutTompkinsModel(rm_ps);
  }
  if(modelType == "NthOrder"){
    d_rateModel = scinew NthOrderModel(rm_ps);
  }
#endif

  //__________________________________
  //  Are we saving the total burned mass and total burned energy
  if ( m_output->isLabelSaved( "totalMassBurned" ) ){
    d_saveConservedVars->mass  = true;
  }

  if ( m_output->isLabelSaved( "totalHeatReleased" ) ){
    d_saveConservedVars->energy = true;
  }

  d_reactant= m_materialManager->parseAndLookupMaterial( srm_ps,"fromMaterial");
  d_product = m_materialManager->parseAndLookupMaterial( srm_ps,"toMaterial");

  //__________________________________
  //  define the materialSet
  d_myMatls = scinew MaterialSet();

  vector<int> m;
  m.push_back(0);                       // needed for the pressure and NC_CCWeight
  m.push_back(d_reactant->getDWIndex());
  m.push_back(d_product->getDWIndex());

  d_myMatls->addAll_unique(m);            // elimiate duplicate entries
  d_myMatls->addReference();
}
//______________________________________________________________________
//
void Ablation0::scheduleInitialize(SchedulerP&,
                                            const LevelP& level)
{
   // None necessary...
}

//______________________________________________________________________
//
void Ablation0::scheduleComputeStableTimeStep(SchedulerP& sched,
                                                       const LevelP& level)
{
   // None necessary...
}
//______________________________________________________________________
//
void Ablation0::scheduleComputeModelSources(SchedulerP& sched,
                                                     const LevelP& level)
{
  Task* t = scinew Task("Ablation0::computeModelSources", this,
                        &Ablation0::computeModelSources);

  printSchedule( level, dout_models_ab0, " Ablation0::scheduleComputeModelSources" );

  Ghost::GhostType  gn  = Ghost::None;
  const MaterialSubset* react_matl = d_reactant->thisMaterial();
  const MaterialSubset* prod_matl  = d_product->thisMaterial();

  MaterialSubset* one_matl     = scinew MaterialSubset();
  one_matl->add(0);
  one_matl->addReference();

  MaterialSubset* press_matl   = one_matl;

  t->requiresVar(Task::OldDW, Ilb->timeStepLabel);
  t->requiresVar(Task::OldDW, Ilb->delTLabel,         level.get_rep());
  //__________________________________
  // Products
  t->requiresVar(Task::NewDW,  Ilb->rho_CCLabel,        prod_matl, gn);
  t->requiresVar(Task::NewDW,  Ilb->specific_heatLabel, prod_matl,gn);

  //__________________________________
  // Reactants
  t->requiresVar(Task::NewDW, Ilb->sp_vol_CCLabel,    react_matl, gn);
  t->requiresVar(Task::OldDW, Ilb->vel_CCLabel,       react_matl, gn);
  t->requiresVar(Task::OldDW, Ilb->temp_CCLabel,      react_matl, gn);
  t->requiresVar(Task::NewDW, Ilb->rho_CCLabel,       react_matl, gn);
  t->requiresVar(Task::NewDW, Ilb->vol_frac_CCLabel,  react_matl, gn);
  t->requiresVar(Task::NewDW, Mlb->pDeltaMassLabel,   react_matl, gn);
  t->requiresVar(Task::OldDW, Mlb->pXLabel,           react_matl, gn);

  t->requiresVar(Task::NewDW, Ilb->press_equil_CCLabel, press_matl,gn);
  

  t->computesVar(reactedFractionLabel, react_matl);
  t->computesVar(delFLabel,            react_matl);

  t->modifiesVar(Ilb->modelMass_srcLabel);
  t->modifiesVar(Ilb->modelMom_srcLabel);
  t->modifiesVar(Ilb->modelEng_srcLabel);
  t->modifiesVar(Ilb->modelVol_srcLabel);

  if(d_saveConservedVars->mass ){
    t->computesVar(Ablation0::totalMassBurnedLabel);
  }
  if(d_saveConservedVars->energy){
    t->computesVar(Ablation0::totalHeatReleasedLabel);
  }
  sched->addTask(t, level->eachPatch(), d_myMatls);

  if (one_matl->removeReference())
    delete one_matl;
}

//______________________________________________________________________
//
void Ablation0::computeModelSources(const ProcessorGroup*,
                                             const PatchSubset* patches,
                                             const MaterialSubset* matls,
                                             DataWarehouse* old_dw,
                                             DataWarehouse* new_dw)
{
  timeStep_vartype timeStep;
  old_dw->get(timeStep, Ilb->timeStepLabel );

  bool isNotInitialTimeStep = (timeStep > 0);

  delt_vartype delT;
  const Level* level = getLevel(patches);
  old_dw->get(delT, Ilb->delTLabel, level);

  int m0 = d_reactant->getDWIndex(); /* reactant material */
  int m1 = d_product->getDWIndex();  /* product material */
  double totalBurnedMass = 0;
  double totalHeatReleased = 0;

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    printTask(patches, patch,dout_models_ab0,"Ablation0::computeModelSources");

    CCVariable<double> mass_src_0;
    CCVariable<double> mass_src_1;
    CCVariable<double> mass_0;
 
    CCVariable<Vector> momentum_src_0;
    CCVariable<Vector> momentum_src_1;
 
    CCVariable<double> energy_src_0;
    CCVariable<double> energy_src_1;
    
    CCVariable<double> sp_vol_src_0;
    CCVariable<double> sp_vol_src_1;

    ParticleSubset* pset = old_dw->getParticleSubset(m0, patch);

    constParticleVariable<double> pDeltaMass;
    constParticleVariable<Point> px;
    new_dw->get(pDeltaMass,               Mlb->pDeltaMassLabel,         pset);
    old_dw->get(px,                       Mlb->pXLabel,                 pset);

    new_dw->getModifiable( mass_src_0,    Ilb->modelMass_srcLabel,  m0,patch);
    new_dw->getModifiable( momentum_src_0,Ilb->modelMom_srcLabel,   m0,patch);
    new_dw->getModifiable( energy_src_0,  Ilb->modelEng_srcLabel,   m0,patch);
    new_dw->getModifiable( sp_vol_src_0,  Ilb->modelVol_srcLabel,   m0,patch);

    new_dw->getModifiable( mass_src_1,    Ilb->modelMass_srcLabel,  m1,patch);
    new_dw->getModifiable( momentum_src_1,Ilb->modelMom_srcLabel,   m1,patch);
    new_dw->getModifiable( energy_src_1,  Ilb->modelEng_srcLabel,   m1,patch);
    new_dw->getModifiable( sp_vol_src_1,  Ilb->modelVol_srcLabel,   m1,patch);

    constCCVariable<double> press_CC;
    constCCVariable<double> cv_reactant;
    constCCVariable<double> rctVolFrac;
    constCCVariable<double> rctTemp;
    constCCVariable<double> rctRho;
    constCCVariable<double> rctSpvol;
    constCCVariable<double> prodRho;
    constCCVariable<double> cv_product;
    
    constCCVariable<Vector> rctvel_CC;
    CCVariable<double> Fr;
    CCVariable<double> delF;

    Vector dx = patch->dCell();
    double cell_vol = dx.x()*dx.y()*dx.z();
    Ghost::GhostType  gn  = Ghost::None;

    //__________________________________
    // Reactant data
    old_dw->get( rctTemp,    Ilb->temp_CCLabel,      m0,patch,gn, 0);
    old_dw->get( rctvel_CC,  Ilb->vel_CCLabel,       m0,patch,gn, 0);
    new_dw->get( rctRho,     Ilb->rho_CCLabel,       m0,patch,gn, 0);
    new_dw->get( rctSpvol,   Ilb->sp_vol_CCLabel,    m0,patch,gn, 0);
    new_dw->get( rctVolFrac, Ilb->vol_frac_CCLabel,  m0,patch,gn, 0);

    new_dw->allocateAndPut(Fr,   reactedFractionLabel, m0,patch);
    new_dw->allocateAndPut(delF, delFLabel,            m0,patch);
    Fr.initialize(0.);
    delF.initialize(0.);

    //__________________________________
    // Product Data,
    new_dw->get(prodRho,       Ilb->rho_CCLabel,        m1,patch,gn, 0);
    new_dw->get(cv_product,    Ilb->specific_heatLabel, m1,patch,gn, 0);
    //__________________________________
    //   Misc.
    new_dw->get(press_CC,      Ilb->press_equil_CCLabel,0,  patch,gn, 0);

    // Get the specific heat, this is the value from the input file
    double cv_rct = -1.0;
    MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial *>(m_materialManager->getMaterial(m0));
    ICEMaterial* ice_matl = dynamic_cast<ICEMaterial *>(m_materialManager->getMaterial(m0));

    if(mpm_matl) {
      cv_rct = mpm_matl->getSpecificHeat();
    }
    else if(ice_matl){
      cv_rct = ice_matl->getSpecificHeat();         // ICE has variable specific heats. -Todd
    }


    // iterate over the particles, reduce the mass at the cell center by
    // the amount of mass removed from the particles in the cell
    for(ParticleSubset::iterator it=pset->begin();it!=pset->end();it++){
      particleIndex idx = *it;

      IntVector c;
      patch->findCell(px[idx],c);
      c=c+IntVector(1,0,0);

      double burnedMass = pDeltaMass[idx];
      
      //__________________________________
      // conservation of mass, momentum and energy
      mass_src_0[c]   -= burnedMass;
      mass_src_1[c]   += burnedMass;
      totalBurnedMass += burnedMass;

      Vector momX        = rctvel_CC[c] * burnedMass;
      momentum_src_0[c] -= momX;
      momentum_src_1[c] += momX;

      double energyX     = cv_rct*rctTemp[c]*burnedMass;
      double releasedHeat= burnedMass * d_E0;
      energy_src_0[c]   -= energyX;
      energy_src_1[c]   += energyX + releasedHeat;
      totalHeatReleased += releasedHeat;

      double createdVolx = burnedMass * rctSpvol[c];
      sp_vol_src_0[c]   -= createdVolx;
      sp_vol_src_1[c]   += createdVolx;
    }  // End loop over particles

    //__________________________________
    //    iterate over the domain
    for (CellIterator iter = patch->getCellIterator();!iter.done();iter++){
      IntVector c = *iter;

      double burnedMass = 0.;;
      double F = prodRho[c]/(rctRho[c]+prodRho[c]);

      if( F >= 0.0 && F < 1.0 ){
        double k  = 1.0; //d_rateConstantModel->getConstant(rctTemp[c]);
        double df = 1.0; //d_rateModel->getDifferentialFractionChange(F);
        delF[c] = 0.; //k * df;
      }

      delF[c] *=delT;
      Fr[c]    = F;
      double rctMass = rctRho[c]*cell_vol;
      double prdMass = prodRho[c]*cell_vol;
      burnedMass = min( delF[c]*(prdMass + rctMass), rctMass);

      //__________________________________
      // conservation of mass, momentum and energy
      mass_src_0[c]   -= burnedMass;
      mass_src_1[c]   += burnedMass;
      totalBurnedMass += burnedMass;

      Vector momX        = rctvel_CC[c] * burnedMass;
      momentum_src_0[c] -= momX;
      momentum_src_1[c] += momX;

      double energyX     = cv_rct*rctTemp[c]*burnedMass;
      double releasedHeat= burnedMass * d_E0;
      energy_src_0[c]   -= energyX;
      energy_src_1[c]   += energyX + releasedHeat;
      totalHeatReleased += releasedHeat;

      double createdVolx = burnedMass * rctSpvol[c];
      sp_vol_src_0[c]   -= createdVolx;
      sp_vol_src_1[c]   += createdVolx;
    }  // cell iterator

    //__________________________________
    //  set symetric BC
    setBC(mass_src_0, "set_if_sym_BC",patch, m_materialManager, m0, new_dw, isNotInitialTimeStep);
    setBC(mass_src_1, "set_if_sym_BC",patch, m_materialManager, m1, new_dw, isNotInitialTimeStep);
    setBC(delF,       "set_if_sym_BC",patch, m_materialManager, m0, new_dw, isNotInitialTimeStep);
    setBC(Fr,         "set_if_sym_BC",patch, m_materialManager, m0, new_dw, isNotInitialTimeStep);
  }

  //__________________________________
  //save total quantities
  if(d_saveConservedVars->mass ){
    new_dw->put(sum_vartype(totalBurnedMass),   Ablation0::totalMassBurnedLabel);
  }
  if(d_saveConservedVars->energy){
    new_dw->put(sum_vartype(totalHeatReleased), Ablation0::totalHeatReleasedLabel);
  }
}
