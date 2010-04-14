/*

The MIT License

Copyright (c) 1997-2010 Center for the Simulation of Accidental Fires and 
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



#include <CCA/Components/Models/HEChem/DDT0.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Material.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Labels/ICELabel.h>
#include <Core/Labels/MPMLabel.h>
#include <Core/Labels/MPMICELabel.h>
#include <CCA/Components/ICE/ICEMaterial.h>
#include <CCA/Components/ICE/BoundaryCond.h>
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <iostream>
#include <Core/Util/DebugStream.h>

using namespace Uintah;
using namespace std;
//__________________________________
//  setenv SCI_DEBUG "MODELS_DOING_COUT:+"
//  MODELS_DOING_COUT:   dumps when tasks are scheduled and performed
static DebugStream cout_doing("MODELS_DOING_COUT", false);

DDT0::DDT0(const ProcessorGroup* myworld,
                         ProblemSpecP& params,
                         const ProblemSpecP& prob_spec)
  : ModelInterface(myworld), d_prob_spec(prob_spec), d_params(params)
{
  d_mymatls = 0;
  Ilb  = scinew ICELabel();
  MIlb = scinew MPMICELabel();
  Mlb  = scinew MPMLabel();
  
  MaterialSubset* d_one_matl = scinew MaterialSubset();
  d_one_matl->add(0);
  d_one_matl->addReference();
  //__________________________________
  //  diagnostic labels JWL++
  reactedFractionLabel   = VarLabel::create("F",
                                      CCVariable<double>::getTypeDescription());
                     
  delFLabel   = VarLabel::create("delF",
                                      CCVariable<double>::getTypeDescription());

  detonatingLabel = VarLabel::create("detonating",
                                      CCVariable<double>::getTypeDescription());
  //__________________________________
  //  diagnostic labels Simple Burn    
  d_saveConservedVars = scinew saveConservedVars();
  onSurfaceLabel   = VarLabel::create("DDT0::onSurface",
                                       CCVariable<double>::getTypeDescription());
    
  surfaceTempLabel = VarLabel::create("DDT0::surfaceTemp",
                                       CCVariable<double>::getTypeDescription());

  burningLabel = VarLabel::create("burning",
                     CCVariable<double>::getTypeDescription());
    
  totalMassBurnedLabel  = VarLabel::create( "totalMassBurned",
                                             sum_vartype::getTypeDescription() );
    
  totalHeatReleasedLabel= VarLabel::create( "totalHeatReleased",
                                             sum_vartype::getTypeDescription() );

}

DDT0::~DDT0()
{
  delete Ilb;
  delete MIlb;
  delete Mlb;

  // JWL++
  VarLabel::destroy(reactedFractionLabel);
  VarLabel::destroy(delFLabel);
  VarLabel::destroy(detonatingLabel);
  // Simple Burn
  VarLabel::destroy(surfaceTempLabel);
  VarLabel::destroy(onSurfaceLabel);
  VarLabel::destroy(burningLabel);
  VarLabel::destroy(totalMassBurnedLabel);
  VarLabel::destroy(totalHeatReleasedLabel);


  if(d_mymatls && d_mymatls->removeReference())
    delete d_mymatls;
    
  if (d_one_matl->removeReference())
    delete d_one_matl;
}

void DDT0::problemSetup(GridP&, SimulationStateP& sharedState,
			     ModelSetup*)
{
  d_sharedState = sharedState;
  bool defaultActive=true;
  // Required for JWL++
  d_params->getWithDefault("Active", d_active, defaultActive);
  d_params->require("ThresholdPressureJWL",   d_threshold_pressure);
  d_params->require("fromMaterial",fromMaterial);
  d_params->require("toMaterial",toMaterial);
  d_params->require("G",    d_G);
  d_params->require("b",    d_b);
  d_params->require("E0",   d_E0);
  d_params->require("rho0", d_rho0);
  d_params->getWithDefault("ThresholdVolFrac",d_threshold_volFrac, 0.01);

  if(d_active){
    // Required for Simple Burn
    d_matl0 = sharedState->parseAndLookupMaterial(d_params, "fromMaterial");
    d_matl1 = sharedState->parseAndLookupMaterial(d_params, "toMaterial");
    d_params->require("Enthalpy",         d_Enthalpy);
    d_params->require("BurnCoeff",        d_BurnCoeff);
    d_params->require("refPressure",      d_refPress);
    d_params->require("ThresholdTemp",    d_thresholdTemp);
    d_params->require("ThresholdPressureSB",d_thresholdPress);

    //__________________________________
    //  define the materialSet
    d_mymatls = scinew MaterialSet();

    vector<int> m;
    m.push_back(0);                                 // needed for the pressure and NC_CCWeight
    m.push_back(d_matl0->getDWIndex());
    m.push_back(d_matl1->getDWIndex());

    d_mymatls->addAll_unique(m);                    // elimiate duplicate entries
    d_mymatls->addReference();
  }

  //__________________________________
  //  Are we saving the total burned mass and total burned energy
  ProblemSpecP DA_ps = d_prob_spec->findBlock("DataArchiver");
  for (ProblemSpecP child = DA_ps->findBlock("save"); child != 0;
                    child = child->findNextBlock("save")) {
    map<string,string> var_attr;
    child->getAttributes(var_attr);
    if (var_attr["label"] == "totalMassBurned"){
      d_saveConservedVars->mass  = true;
    }
    if (var_attr["label"] == "totalHeatReleased"){
      d_saveConservedVars->energy = true;
    }
  }
}

//______________________________________________________________________
//
void DDT0::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP model_ps = ps->appendChild("Model");
  model_ps->setAttribute("type","DDT0");

  model_ps->appendElement("Active",d_active);
  model_ps->appendElement("ThresholdPressureJWL",d_threshold_pressure);
  model_ps->appendElement("fromMaterial",fromMaterial);
  model_ps->appendElement("toMaterial",toMaterial);
  model_ps->appendElement("G",    d_G);
  model_ps->appendElement("b",    d_b);
  model_ps->appendElement("E0",   d_E0);
  model_ps->appendElement("rho0", d_rho0);
  model_ps->appendElement("ThresholdTemp",     d_thresholdTemp);
  model_ps->appendElement("ThresholdPressureSB", d_thresholdPress);
  model_ps->appendElement("ThresholdVolFrac", d_threshold_volFrac);
  model_ps->appendElement("fromMaterial",      d_matl0->getName());
  if(d_active){
    model_ps->appendElement("toMaterial",        d_matl1->getName());
    model_ps->appendElement("Enthalpy",          d_Enthalpy);
    model_ps->appendElement("BurnCoeff",         d_BurnCoeff);
    model_ps->appendElement("refPressure",       d_refPress);
  }
}

//______________________________________________________________________
//
void DDT0::activateModel(GridP&, SimulationStateP& sharedState, ModelSetup*)
{
  d_active=true;
#if 0
  d_params->require("G",    d_G);
  d_params->require("b",    d_b);
  d_params->require("E0",   d_E0);
  d_params->require("rho0", d_rho0);
#endif
  d_params->require("Enthalpy",         d_Enthalpy);
  d_params->require("BurnCoeff",        d_BurnCoeff);
  d_params->require("refPressure",      d_refPress);

  d_matl0 = sharedState->parseAndLookupMaterial(d_params, "fromMaterial");
  d_matl1 = sharedState->parseAndLookupMaterial(d_params, "toMaterial");
 
  
  //__________________________________
  //  define the materialSet
  d_mymatls = scinew MaterialSet();

  vector<int> m;
  m.push_back(0);                                 // needed for the pressure and NC_CCWeight
  m.push_back(d_matl0->getDWIndex());
  m.push_back(d_matl1->getDWIndex());

  d_mymatls->addAll_unique(m);                    // elimiate duplicate entries
  d_mymatls->addReference();
}

//______________________________________________________________________
//     
void DDT0::scheduleInitialize(SchedulerP& sched,
                               const LevelP& level,
                               const ModelInfo*)
{
  printSchedule(level,"DDT0::scheduleInitialize\t\t\t");
  Task* t = scinew Task("DDT0::initialize", this, &DDT0::initialize);
  const MaterialSubset* react_matl = d_matl0->thisMaterial();
  t->computes(reactedFractionLabel, react_matl);
  t->computes(burningLabel,         react_matl);
  sched->addTask(t, level->eachPatch(), d_mymatls);
}

//______________________________________________________________________
//
void DDT0::initialize(const ProcessorGroup*,
                             const PatchSubset* patches,
                             const MaterialSubset* /*matls*/,
                             DataWarehouse*,
                             DataWarehouse* new_dw){
  int m0 = d_matl0->getDWIndex();
  for(int p=0;p<patches->size();p++) {
    const Patch* patch = patches->get(p);
    cout_doing << "Doing Initialize on patch " << patch->getID()<< "\t\t\t STEADY_BURN" << endl;
    
    // This section is needed for outputting F and burn on each timestep
    CCVariable<double> F, burn;
    new_dw->allocateAndPut(F, reactedFractionLabel, m0, patch);
    new_dw->allocateAndPut(burn, burningLabel,      m0, patch);

    F.initialize(0.0);
    burn.initialize(0.0);
  }
}

//______________________________________________________________________
//      
void DDT0::scheduleComputeStableTimestep(SchedulerP&,
                                          const LevelP&,
                                          const ModelInfo*)
{
  // None necessary...
}

//______________________________________________________________________
//     
void DDT0::scheduleComputeModelSources(SchedulerP& sched,
                                       const LevelP& level,
                                       const ModelInfo* mi)
{
  if(d_active){
    Task* t = scinew Task("DDT0::computeModelSources", this, 
                          &DDT0::computeModelSources, mi);
    cout_doing << "DDT0::scheduleComputeModelSources "<<  endl;  
    Ghost::GhostType  gac = Ghost::AroundCells;
    Ghost::GhostType  gn  = Ghost::None;
    const MaterialSubset* react_matl = d_matl0->thisMaterial();
    const MaterialSubset* prod_matl  = d_matl1->thisMaterial();

    const MaterialSet* all_matls = d_sharedState->allMaterials();
    const MaterialSubset* all_matls_sub = all_matls->getUnion();
    Task::DomainSpec oms = Task::OutOfDomain;

    //__________________________________
    // Requires
    //__________________________________
    t->requires( Task::OldDW, mi->delT_Label,       level.get_rep());
    t->requires(Task::OldDW, Ilb->temp_CCLabel,     all_matls_sub, oms, gac,1);
    t->requires(Task::NewDW, Ilb->vol_frac_CCLabel, all_matls_sub, oms, gac,1);
    

    //__________________________________
    // Products
    t->requires(Task::NewDW,  Ilb->rho_CCLabel,     prod_matl, gn);
    t->requires(Task::OldDW,  Ilb->temp_CCLabel,    prod_matl, gn);       
    t->requires(Task::NewDW,  Ilb->vol_frac_CCLabel,prod_matl, gn);       
    t->requires(Task::NewDW,  Ilb->TempX_FCLabel,   prod_matl, gac,2);    
    t->requires(Task::NewDW,  Ilb->TempY_FCLabel,   prod_matl, gac,2);    
    t->requires(Task::NewDW,  Ilb->TempZ_FCLabel,   prod_matl, gac,2);
    t->requires(Task::NewDW,  Ilb->press_equil_CCLabel, d_one_matl,  gn);
    t->requires(Task::OldDW,  MIlb->NC_CCweightLabel,   d_one_matl,  gac, 1);
        
  
    //__________________________________
    // Reactants
    t->requires(Task::NewDW, Ilb->sp_vol_CCLabel,   react_matl, gn);
    t->requires(Task::OldDW, MIlb->vel_CCLabel,     react_matl, gn);
    t->requires(Task::OldDW, MIlb->temp_CCLabel,    react_matl, gn);
    t->requires(Task::NewDW, MIlb->temp_CCLabel,    react_matl, gn);
    t->requires(Task::NewDW, Ilb->rho_CCLabel,      react_matl, gn);
    t->requires(Task::OldDW, Mlb->pXLabel,          react_matl, gn);
    t->requires(Task::NewDW, MIlb->cMassLabel,      react_matl, gn);
    t->requires(Task::NewDW, Mlb->gMassLabel,       react_matl, gac,1); 
    t->requires(Task::OldDW, reactedFractionLabel,  react_matl, gn); 
    t->requires(Task::NewDW, Ilb->vol_frac_CCLabel,  react_matl, gn); 

    //__________________________________
    // Computes
    //__________________________________
    t->computes(reactedFractionLabel, react_matl);
    t->computes(delFLabel,            react_matl);
    t->computes(burningLabel,         react_matl);
    t->computes(detonatingLabel,      react_matl);
    t->computes(DDT0::onSurfaceLabel,    react_matl);
    t->computes(DDT0::surfaceTempLabel,  react_matl);

    //__________________________________
    // Conserved Variables
    //__________________________________
    if(d_saveConservedVars->mass ){
        t->computes(DDT0::totalMassBurnedLabel);
    }
    if(d_saveConservedVars->energy){
        t->computes(DDT0::totalHeatReleasedLabel);
    }
    
    //__________________________________
    // Modifies  
    //__________________________________
    t->modifies(mi->modelMass_srcLabel);
    t->modifies(mi->modelMom_srcLabel);
    t->modifies(mi->modelEng_srcLabel);
    t->modifies(mi->modelVol_srcLabel); 
    sched->addTask(t, level->eachPatch(), d_mymatls);
  }
}

//______________________________________________________________________
void DDT0::scheduleCheckNeedAddMaterial(SchedulerP& sched,
                                         const LevelP& level,
                                         const ModelInfo* mi)
{
    Task* t = scinew Task("DDT0::checkNeedAddMaterial", this, 
                          &DDT0::checkNeedAddMaterial, mi);
    cout_doing << "DDT0::scheduleCheckNeedAddMaterial "<<  endl;  

    Ghost::GhostType  gn  = Ghost::None;
    Ghost::GhostType  gac = Ghost::AroundCells;

    const MaterialSubset* react_matl = d_matl0->thisMaterial();


    t->requires(Task::NewDW, Ilb->press_equil_CCLabel, d_one_matl, gn);
    t->requires(Task::OldDW, MIlb->NC_CCweightLabel,   d_one_matl, gac,1);
    t->requires(Task::NewDW, Mlb->gMassLabel,          react_matl, gac,1);
    t->requires(Task::NewDW, MIlb->temp_CCLabel,       react_matl, gn);
    
    t->computes(DDT0::surfaceTempLabel,   d_one_matl);
    t->computes(Ilb->NeedAddIceMaterialLabel);

    sched->addTask(t, level->eachPatch(), d_mymatls);
}

//______________________________________________________________________
void
DDT0::checkNeedAddMaterial(const ProcessorGroup*,
                            const PatchSubset* patches,
                            const MaterialSubset*,
                            DataWarehouse* old_dw,
                            DataWarehouse* new_dw,
                            const ModelInfo* /*mi*/)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);  
    
    cout_doing << "Doing checkNeedAddMaterial on patch "<< patch->getID()
               <<"\t\t\t\t  DDT0" << endl;

    Ghost::GhostType  gn  = Ghost::None;
    Ghost::GhostType  gac = Ghost::AroundCells;
    int m0 = d_matl0->getDWIndex();

    constCCVariable<double> solidTemp;
    CCVariable<double> surfaceTemp;
    constNCVariable<double> NC_CCweight,NCsolidMass;
    constCCVariable<Vector> vel_CC;
    constCCVariable<double> press_CC;
    new_dw->get(press_CC,   Ilb->press_equil_CCLabel,0,  patch,gn, 0);
      
    //__________________________________
    // Reactant data
    new_dw->get(solidTemp,   Ilb->temp_CCLabel, m0,patch,gn, 0);
    new_dw->get(NCsolidMass, Mlb->gMassLabel,   m0,patch,gac,1);
      
    //__________________________________
    //   Misc.
    old_dw->get(NC_CCweight,     MIlb->NC_CCweightLabel,   0,  patch,gac,1);   
    new_dw->allocateAndPut(surfaceTemp,DDT0::surfaceTempLabel, 0, patch);
    surfaceTemp.initialize(0.);
      
    IntVector nodeIdx[8];

    // JWL++ logic
    double need_add=0.;
    if(!d_active){
      bool add = false;
      for (CellIterator iter = patch->getCellIterator();!iter.done();iter++){
        IntVector c = *iter;
        if (press_CC[c] > .9*d_threshold_pressure){
          add = true;
        }
                  
        //__________________________________
        // Find if the cell contains surface:
        patch->findNodesFromCell(*iter,nodeIdx);
        double MaxMass = d_SMALL_NUM;
        double MinMass = 1.0/d_SMALL_NUM;
        for (int nN=0; nN<8; nN++) {
            MaxMass = std::max(MaxMass,NC_CCweight[nodeIdx[nN]]*
                                NCsolidMass[nodeIdx[nN]]);
            MinMass = std::min(MinMass,NC_CCweight[nodeIdx[nN]]*
                                NCsolidMass[nodeIdx[nN]]); 
      }               
          
      if ( (MaxMass-MinMass)/MaxMass > 0.4       // Find the "surface"
           && (MaxMass-MinMass)/MaxMass < 1.0
           &&  MaxMass > d_TINY_RHO){
              
          //__________________________________
          //  On the surface, determine the maxiumum temperature
          //  use this to determine if it is time to activate the model.
          surfaceTemp[c] = solidTemp[c];
          if(surfaceTemp[c] > .95*d_thresholdTemp){
              need_add=1.;
          }
       }  // if (maxMass-MinMass....)
    }

     if(add){
       need_add=1.;
     }
     else{
       need_add=0.;
     }
    }  //only add a new material once
    else{
      need_add=0.;
    }
      
    new_dw->put(sum_vartype(need_add),     Ilb->NeedAddIceMaterialLabel);
  }
}
//______________________________________________________________________
//
void DDT0::computeModelSources(const ProcessorGroup*, 
                                const PatchSubset* patches,
                                const MaterialSubset*,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw,
                                const ModelInfo* mi)
{
  delt_vartype delT;
  const Level* level = getLevel(patches);
  old_dw->get(delT, mi->delT_Label, level);

 
  int m0 = d_matl0->getDWIndex();
  int m1 = d_matl1->getDWIndex();
  double totalBurnedMass = 0;
  double totalHeatReleased = 0;
 
  double massConv = 0;


  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);  
    
    cout_doing << "Doing computeModelSources on patch "<< patch->getID()
               <<"\t\t\t\t  DDT0" << endl;

    // Variable to modify
    CCVariable<double> mass_src_0, mass_src_1, mass_0;
    CCVariable<Vector> momentum_src_0, momentum_src_1;
    CCVariable<double> energy_src_0, energy_src_1;
    CCVariable<double> sp_vol_src_0, sp_vol_src_1;
    CCVariable<double> onSurface, surfaceTemp;

    new_dw->getModifiable(mass_src_0,    mi->modelMass_srcLabel,  m0,patch);
    new_dw->getModifiable(momentum_src_0,mi->modelMom_srcLabel,   m0,patch);
    new_dw->getModifiable(energy_src_0,  mi->modelEng_srcLabel,   m0,patch);
    new_dw->getModifiable(sp_vol_src_0,  mi->modelVol_srcLabel,   m0,patch);

    new_dw->getModifiable(mass_src_1,    mi->modelMass_srcLabel,  m1,patch);
    new_dw->getModifiable(momentum_src_1,mi->modelMom_srcLabel,   m1,patch);
    new_dw->getModifiable(energy_src_1,  mi->modelEng_srcLabel,   m1,patch);
    new_dw->getModifiable(sp_vol_src_1,  mi->modelVol_srcLabel,   m1,patch);

    // New Variables to store for this timestep
    constCCVariable<double> press_CC, cv_reactant, gasTemp,gasVol_frac,solidTemp,solidMass, rctVolFrac;
    constCCVariable<double> rctTemp,rctRho,rctSpvol,prodRho, rctFr;
    constCCVariable<Vector> rctvel_CC;
    CCVariable<double> Fr;
    CCVariable<double> delF;
    CCVariable<double> burning, detonating;
    constNCVariable<double> NC_CCweight,NCsolidMass;
    constSFCXVariable<double> gasTempX_FC;
    constSFCYVariable<double> gasTempY_FC;
    constSFCZVariable<double> gasTempZ_FC;
	    
    Vector dx = patch->dCell();
    double delX = dx.x();
    double delY = dx.y();
    double cell_vol = dx.x()*dx.y()*dx.z();
    Ghost::GhostType  gn  = Ghost::None;
    Ghost::GhostType  gac = Ghost::AroundCells;
   
    //__________________________________
    // Reactant data
    old_dw->get(rctTemp,       MIlb->temp_CCLabel,  m0,patch,gn, 0);
    old_dw->get(rctvel_CC,     MIlb->vel_CCLabel,   m0,patch,gn, 0);
    new_dw->get(rctRho,        Ilb->rho_CCLabel,    m0,patch,gn, 0);
    new_dw->get(rctSpvol,      Ilb->sp_vol_CCLabel, m0,patch,gn, 0);
    new_dw->get(solidTemp,     MIlb->temp_CCLabel,  m0,patch,gn, 0);
    new_dw->get(solidMass,     MIlb->cMassLabel,    m0,patch,gn, 0);
    new_dw->get(NCsolidMass,   Mlb->gMassLabel,     m0,patch,gac,1);
    //new_dw->get(cv_reactant,   Ilb->specific_heatLabel,m0,patch,gn, 0);
    old_dw->get(rctFr,         reactedFractionLabel,  m0,patch,gac,0);
    new_dw->get(rctVolFrac,    Ilb->vol_frac_CCLabel,  m0,patch,gn, 0);
    new_dw->allocateAndPut(Fr,   reactedFractionLabel,m0,patch);
    new_dw->allocateAndPut(delF, delFLabel,           m0,patch);
    Fr.initialize(0.);
    delF.initialize(0.);
    //__________________________________
    // Product Data, 
    new_dw->get(prodRho,         Ilb->rho_CCLabel,   m1,patch,gn, 0);
    new_dw->get(gasTempX_FC,      Ilb->TempX_FCLabel,m1,patch,gac,2);
    new_dw->get(gasTempY_FC,      Ilb->TempY_FCLabel,m1,patch,gac,2);
    new_dw->get(gasTempZ_FC,      Ilb->TempZ_FCLabel,m1,patch,gac,2);
    old_dw->get(gasTemp,          Ilb->temp_CCLabel, m1,patch,gn, 0);
    new_dw->get(gasVol_frac,      Ilb->vol_frac_CCLabel,  m1, patch,gn, 0);
    //__________________________________
    //   Misc.
    new_dw->get(press_CC,         Ilb->press_equil_CCLabel,0,  patch,gn, 0);
    old_dw->get(NC_CCweight,      MIlb->NC_CCweightLabel,  0,   patch,gac,1);   
    
    new_dw->allocateAndPut(burning,    burningLabel,           m0, patch,gn,0);  
    new_dw->allocateAndPut(detonating, detonatingLabel,        m0, patch);
    burning.initialize(0.);
    detonating.initialize(0.);

    new_dw->allocateAndPut(onSurface,  DDT0::onSurfaceLabel,   m0, patch);
    new_dw->allocateAndPut(surfaceTemp,DDT0::surfaceTempLabel, m0, patch);

    IntVector nodeIdx[8];
      
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m0);
    double cv_solid = mpm_matl->getSpecificHeat();
     
    // Get Temperatures for burning check
    int numAllMatls = d_sharedState->getNumMatls();
    StaticArray<constCCVariable<double> > vol_frac_CC(numAllMatls);
    StaticArray<constCCVariable<double> > temp_CC(numAllMatls);
    for(int m = 0; m < numAllMatls; m++) {
      Material* matl = d_sharedState->getMaterial(m);
      int indx = matl->getDWIndex();
      old_dw->get(temp_CC[m],     MIlb->temp_CCLabel,    indx, patch, gac, 1);
      new_dw->get(vol_frac_CC[m], Ilb->vol_frac_CCLabel, indx, patch, gac, 1);
    }

    for (CellIterator iter = patch->getCellIterator();!iter.done();iter++){
      IntVector c = *iter;
      // Copy old Fr in case pressure threshold is not met
      Fr[c]=rctFr[c];

      // JWL++ Model For explosions
      if (press_CC[c] > d_threshold_pressure && rctVolFrac[c] > d_threshold_volFrac){
        // Flag for detonating
        detonating[c] = 1;

        double burnedMass;

        double F = prodRho[c]/(rctRho[c]+prodRho[c]);
        if(F >= 0. && F < .99){
          delF[c] = d_G*pow(press_CC[c],d_b)*(1.-F);
        }
        delF[c]*=delT;
        Fr[c] = F;
        double rctMass = rctRho[c]*cell_vol;
        double prdMass = prodRho[c]*cell_vol;
        burnedMass = min(delF[c]*(prdMass+rctMass), rctMass);
        burnedMass = min(burnedMass, .2*d_rho0*cell_vol);

        //__________________________________
        // conservation of mass, momentum and energy                           
        mass_src_0[c] -= burnedMass;
        mass_src_1[c] += burnedMass;
        massConv += burnedMass;           

        Vector momX        = rctvel_CC[c] * burnedMass;
        momentum_src_0[c] -= momX;
        momentum_src_1[c] += momX;
      
        double energyX   = dynamic_cast<const MPMMaterial*>(d_matl0)->getSpecificHeat()*rctTemp[c]*burnedMass;//*cv_reactant[c]; 
        double releasedHeat = burnedMass * d_E0;
        energy_src_0[c] -= energyX;
        energy_src_1[c] += energyX + releasedHeat;

        double createdVolx  = burnedMass * rctSpvol[c];
        sp_vol_src_0[c] -= createdVolx;
        sp_vol_src_1[c] += createdVolx;
      } else if(press_CC[c] < d_threshold_pressure && press_CC[c] > d_thresholdPress) {
        //__________________________________
        // Find if the cell contains surface:
        patch->findNodesFromCell(*iter,nodeIdx);
        double MaxMass = d_SMALL_NUM;
        double MinMass = 1.0/d_SMALL_NUM;                 
        for (int nN=0; nN<8; nN++) {
            MaxMass = std::max(MaxMass,NC_CCweight[nodeIdx[nN]]*
                               NCsolidMass[nodeIdx[nN]]);
            MinMass = std::min(MinMass,NC_CCweight[nodeIdx[nN]]*
                               NCsolidMass[nodeIdx[nN]]); 
        }               
          
        //===============================================
        //If you change the burning criteria logic you must also modify
        //CCA/Components/SwitchCriteria
        //===============================================
        if ( (MaxMass-MinMass)/MaxMass > 0.4            //--------------KNOB 1
            && (MaxMass-MinMass)/MaxMass < 1.0
            &&  MaxMass > d_TINY_RHO){



            //__________________________________
            //  Determine the temperature
            //  to use in burn model
            double Temp = 0;
            
            for (int m = 0; m < numAllMatls; m++) {
              if(vol_frac_CC[m][c] > 0.2 && temp_CC[m][c] > d_thresholdTemp && temp_CC[m][c] > Temp )
                Temp = temp_CC[m][c];
            }  
            
            surfaceTemp[c] = Temp;
              
            double surfArea = delX*delY;  
            onSurface[c] = surfArea; // debugging var
              
            //__________________________________
            //  Simple Burn Model
            double burnedMass = 0.0;
            if ((Temp > d_thresholdTemp) && (press_CC[c] > d_thresholdPress)) {
                // Flag for burning
                burning[c] = 1;
                burnedMass = delT *surfArea * d_BurnCoeff 
                * pow((press_CC[c]/d_refPress),0.778);
            }
            if(burnedMass > solidMass[c]){
                burnedMass = solidMass[c];
            }
              
            //__________________________________
            // conservation of mass, momentum and energy                           
            mass_src_0[c] -= burnedMass;
            mass_src_1[c] += burnedMass;
            totalBurnedMass += burnedMass;
              
            Vector momX        = rctvel_CC[c] * burnedMass;
            momentum_src_0[c] -= momX;
            momentum_src_1[c] += momX;
 
            double energyX   = cv_solid*solidTemp[c]*burnedMass; 
            double releasedHeat = burnedMass * d_Enthalpy;
            energy_src_0[c] -= energyX;
            energy_src_1[c] += energyX + releasedHeat;
            totalHeatReleased += releasedHeat;

            double createdVolx  = burnedMass * rctSpvol[c];
            sp_vol_src_0[c] -= createdVolx;
            sp_vol_src_1[c] += createdVolx;
        }  // if (maxMass-MinMass....)
      }// if (pressure)
    }  // cell iterator  

    //__________________________________
    //  set symetric BC
    setBC(mass_src_0, "set_if_sym_BC",patch, d_sharedState, m0, new_dw);
    setBC(mass_src_1, "set_if_sym_BC",patch, d_sharedState, m1, new_dw);
    setBC(delF,       "set_if_sym_BC",patch, d_sharedState, m0, new_dw);
    setBC(Fr,         "set_if_sym_BC",patch, d_sharedState, m0, new_dw);
  }
  
  //__________________________________
  //save total quantities
  if(d_saveConservedVars->mass ){
      new_dw->put(sum_vartype(totalBurnedMass),  DDT0::totalMassBurnedLabel);
  }
  if(d_saveConservedVars->energy){
      new_dw->put(sum_vartype(totalHeatReleased),DDT0::totalHeatReleasedLabel);
  }
}
//______________________________________________________________________
//
void DDT0::scheduleModifyThermoTransportProperties(SchedulerP&,
                                                    const LevelP&,
                                                    const MaterialSet*)
{
  // do nothing      
}
void DDT0::computeSpecificHeat(CCVariable<double>&,
                                const Patch*,   
                                DataWarehouse*, 
                                const int)      
{
  //do nothing
}
//______________________________________________________________________
//
void DDT0::scheduleErrorEstimate(const LevelP&,
                                  SchedulerP&)
{
  // Not implemented yet
}
//__________________________________
void DDT0::scheduleTestConservation(SchedulerP&,
                                     const PatchSet*,                      
                                     const ModelInfo*)                     
{
  // Not implemented yet
}

//______________________________________________________________________
//
void DDT0::printSchedule(const LevelP& level,
                                const string& where){
  if (cout_doing.active()){
    cout_doing << d_myworld->myrank() << " "
               << where << "L-"
               << level->getIndex()<< endl;
  }
}

