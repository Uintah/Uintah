/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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



#include <CCA/Components/ICE/BoundaryCond.h>
#include <CCA/Components/ICE/ICEMaterial.h>
#include <CCA/Components/Models/HEChem/Common.h>
#include <CCA/Components/Models/HEChem/DDT0.h>
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Material.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/DbgOutput.h>
#include <Core/Labels/ICELabel.h>
#include <Core/Labels/MPMICELabel.h>
#include <Core/Labels/MPMLabel.h>
#include <Core/Parallel/Parallel.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Util/DebugStream.h>
#include <iostream>


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
  d_mymatls  = 0;
  d_one_matl = 0;
  Ilb  = scinew ICELabel();
  MIlb = scinew MPMICELabel();
  Mlb  = scinew MPMLabel();
  

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
  onSurfaceLabel   = VarLabel::create("onSurface",
                                       CCVariable<double>::getTypeDescription());
    
  surfaceTempLabel = VarLabel::create("surfaceTemp",
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
    
  if (d_one_matl && d_one_matl->removeReference())
    delete d_one_matl;
}

void DDT0::problemSetup(GridP&, SimulationStateP& sharedState,
			     ModelSetup*)
{
  d_sharedState = sharedState;
  
  // Required for JWL++
  d_params->require("ThresholdPressureJWL",   d_threshold_press_JWL);
  d_params->require("fromMaterial",fromMaterial);
  d_params->require("toMaterial",toMaterial);
  d_params->require("G",    d_G);
  d_params->require("b",    d_b);
  d_params->require("E0",   d_E0);
  d_params->getWithDefault("ThresholdVolFrac",d_threshold_volFrac, 0.01);

  // Required for Simple Burn
  d_matl0 = sharedState->parseAndLookupMaterial(d_params, "fromMaterial");
  d_matl1 = sharedState->parseAndLookupMaterial(d_params, "toMaterial");
  d_params->require("Enthalpy",         d_Enthalpy);
  d_params->require("BurnCoeff",        d_BurnCoeff);
  d_params->require("refPressure",      d_refPress);
  d_params->require("ThresholdTemp",    d_thresholdTemp);
  d_params->require("ThresholdPressureSB",d_thresholdPress_SB);
  d_params->getWithDefault("useCrackModel",    d_useCrackModel, false); 
  
  if(d_useCrackModel){
    d_params->require("Gcrack",           d_Gcrack);
    d_params->getWithDefault("CrackVolThreshold",     d_crackVolThreshold, 1e-14 );
    d_params->require("nCrack",           d_nCrack);
      
    pCrackRadiusLabel = VarLabel::find("p.crackRad");
    if(!pCrackRadiusLabel){
      ostringstream msg;
      msg << "\n ERROR:Model:DDT0: The constitutive model for the MPM reactant must be visco_scram in order to burn in cracks. \n";
      msg << " No other constitutive models are currently supported "; 
      throw ProblemSetupException(msg.str(),__FILE__, __LINE__);
    }
  }

  //__________________________________
  //  define the materialSet
  d_mymatls = scinew MaterialSet();

  vector<int> m;
  m.push_back(0);                                 // needed for the pressure and NC_CCWeight
  m.push_back(d_matl0->getDWIndex());
  m.push_back(d_matl1->getDWIndex());

  d_mymatls->addAll_unique(m);                    // elimiate duplicate entries
  d_mymatls->addReference();

  d_one_matl = scinew MaterialSubset();
  d_one_matl->add(0);
  d_one_matl->addReference();

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

  model_ps->appendElement("ThresholdPressureJWL",d_threshold_press_JWL);
  model_ps->appendElement("fromMaterial",fromMaterial);
  model_ps->appendElement("toMaterial",  toMaterial);
  model_ps->appendElement("G",    d_G);
  model_ps->appendElement("b",    d_b);
  model_ps->appendElement("E0",   d_E0);

  model_ps->appendElement("ThresholdTemp",       d_thresholdTemp);
  model_ps->appendElement("ThresholdPressureSB", d_thresholdPress_SB);
  model_ps->appendElement("ThresholdVolFrac",    d_threshold_volFrac);
  model_ps->appendElement("fromMaterial",        d_matl0->getName());
  model_ps->appendElement("toMaterial",          d_matl1->getName());
  model_ps->appendElement("Enthalpy",            d_Enthalpy);   
  model_ps->appendElement("BurnCoeff",           d_BurnCoeff);  
  model_ps->appendElement("refPressure",         d_refPress);  
  if(d_useCrackModel){
    model_ps->appendElement("useCrackModel",     d_useCrackModel);
    model_ps->appendElement("Gcrack",            d_Gcrack);
    model_ps->appendElement("nCrack",            d_nCrack);
    model_ps->appendElement("CrackVolThreshold", d_crackVolThreshold);
  }
}

//______________________________________________________________________
//     
void DDT0::scheduleInitialize(SchedulerP& sched,
                               const LevelP& level,
                               const ModelInfo*)
{
  printSchedule(level,cout_doing,"DDT0::scheduleInitialize");
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
  if (level->hasFinerLevel()){
    return;    // only schedule on the finest level
  }

  Task* t = scinew Task("DDT0::computeModelSources", this, 
                        &DDT0::computeModelSources, mi);
  cout_doing << "DDT0::scheduleComputeModelSources "<<  endl;  
  Ghost::GhostType  gac = Ghost::AroundCells;
  Ghost::GhostType  gn  = Ghost::None;
  const MaterialSubset* react_matl = d_matl0->thisMaterial();
  const MaterialSubset* prod_matl  = d_matl1->thisMaterial();

  const MaterialSubset* all_matls = d_sharedState->allMaterials()->getUnion();
  const MaterialSubset* ice_matls = d_sharedState->allICEMaterials()->getUnion();
  const MaterialSubset* mpm_matls = d_sharedState->allMPMMaterials()->getUnion();
  Task::MaterialDomainSpec oms = Task::OutOfDomain;

  proc0cout << "\nDDT0:scheduleComputeModelSources oneMatl " << *d_one_matl<< " react_matl " << *react_matl 
                                               << " prod_matl " << *prod_matl 
                                               << " all_matls " << *all_matls 
                                               << " ice_matls " << *ice_matls <<" mpm_matls " << *mpm_matls << "\n"<<endl;
  //__________________________________
  // Requires
  //__________________________________
  t->requires(Task::OldDW, mi->delT_Label,        level.get_rep());
  t->requires(Task::OldDW, Ilb->temp_CCLabel,     ice_matls, oms, gn);
  t->requires(Task::NewDW, Ilb->temp_CCLabel,     mpm_matls, oms, gn);
  t->requires(Task::NewDW, Ilb->vol_frac_CCLabel, all_matls, oms, gn);
  if(d_useCrackModel){
    t->requires(Task::OldDW, Mlb->pXLabel,        mpm_matls,  gn);
    t->requires(Task::OldDW, pCrackRadiusLabel,   react_matl, gn);
  }

  //__________________________________
  // Products
  t->requires(Task::NewDW,  Ilb->rho_CCLabel,         prod_matl,   gn); 
  t->requires(Task::NewDW,  Ilb->press_equil_CCLabel, d_one_matl,  gn);
  t->requires(Task::OldDW,  MIlb->NC_CCweightLabel,   d_one_matl,  gac, 1);

  //__________________________________
  // Reactants
  t->requires(Task::NewDW, Ilb->sp_vol_CCLabel,   react_matl, gn);
  t->requires(Task::NewDW, MIlb->vel_CCLabel,     react_matl, gn);
  t->requires(Task::NewDW, Ilb->rho_CCLabel,      react_matl, gn);
  t->requires(Task::NewDW, Mlb->gMassLabel,       react_matl, gac,1);

  //__________________________________
  // Computes
  //__________________________________
  t->computes(reactedFractionLabel,    react_matl);
  t->computes(delFLabel,               react_matl);
  t->computes(burningLabel,            react_matl);
  t->computes(detonatingLabel,         react_matl);
  t->computes(onSurfaceLabel,    react_matl);
  t->computes(surfaceTempLabel,  react_matl);

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
  int numAllMatls = d_sharedState->getNumMatls();

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);  
    ParticleSubset* pset = old_dw->getParticleSubset(m0, patch); 
    
    cout_doing << "Doing computeModelSources on patch "<< patch->getID()
               <<"\t\t\t\t  DDT0" << endl;

    // Variable to modify or compute
    CCVariable<double> mass_src_0, mass_src_1, mass_0;
    CCVariable<Vector> momentum_src_0, momentum_src_1;
    CCVariable<double> energy_src_0, energy_src_1;
    CCVariable<double> sp_vol_src_0, sp_vol_src_1;
    CCVariable<double> onSurface, surfaceTemp;
    CCVariable<int>    crackedEnough;
    CCVariable<double> Fr;
    CCVariable<double> delF;
    CCVariable<double> burning, detonating;

    constCCVariable<double> press_CC, cv_reactant, rctVolFrac;
    constCCVariable<double> rctTemp, rctRho, rctSpvol, prodRho, rctFr;
    constCCVariable<Vector> rctvel_CC;
    constNCVariable<double> NC_CCweight, rctMass_NC;
    constParticleVariable<Point> px;

    // Stores level of cracking in particles
    constParticleVariable<double> crackRad;   
 
    StaticArray<constCCVariable<double> > vol_frac_CC(numAllMatls);
    StaticArray<constCCVariable<double> > temp_CC(numAllMatls);
	    
    Vector dx = patch->dCell();
    double cell_vol = dx.x()*dx.y()*dx.z();
    Ghost::GhostType  gn  = Ghost::None;
    Ghost::GhostType  gac = Ghost::AroundCells;
   
    //__________________________________
    // Reactant data
    new_dw->get(rctTemp,       MIlb->temp_CCLabel,    m0,patch,gn, 0);
    new_dw->get(rctvel_CC,     MIlb->vel_CCLabel,     m0,patch,gn, 0);
    new_dw->get(rctRho,        Ilb->rho_CCLabel,      m0,patch,gn, 0);
    new_dw->get(rctSpvol,      Ilb->sp_vol_CCLabel,   m0,patch,gn, 0);
    new_dw->get(rctMass_NC,    Mlb->gMassLabel,       m0,patch,gac,1);
    new_dw->get(rctVolFrac,    Ilb->vol_frac_CCLabel, m0,patch,gn, 0);
    if(d_useCrackModel){
      old_dw->get(px,          Mlb->pXLabel,      pset);
      old_dw->get(crackRad,    pCrackRadiusLabel, pset); 
    }
 
    //__________________________________
    // Product Data, 
    new_dw->get(prodRho,         Ilb->rho_CCLabel,   m1,patch,gn, 0);
    
    //__________________________________
    //   Misc.
    new_dw->get(press_CC,         Ilb->press_equil_CCLabel,0,  patch,gn, 0);
    old_dw->get(NC_CCweight,      MIlb->NC_CCweightLabel,  0,  patch,gac,1);   
    
    for(int m = 0; m < numAllMatls; m++) {
      Material* matl = d_sharedState->getMaterial(m);
      ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
      int indx = matl->getDWIndex();
      if(ice_matl){
        old_dw->get(temp_CC[m],   MIlb->temp_CCLabel,    indx, patch,gn,0);
      }else {
        new_dw->get(temp_CC[m],   MIlb->temp_CCLabel,    indx, patch,gn,0);
      }
      new_dw->get(vol_frac_CC[m], Ilb->vol_frac_CCLabel, indx, patch,gn,0);
    }
   
    //__________________________________
    //  What is computed
    new_dw->getModifiable(mass_src_0,    mi->modelMass_srcLabel,  m0,patch);
    new_dw->getModifiable(momentum_src_0,mi->modelMom_srcLabel,   m0,patch);
    new_dw->getModifiable(energy_src_0,  mi->modelEng_srcLabel,   m0,patch);
    new_dw->getModifiable(sp_vol_src_0,  mi->modelVol_srcLabel,   m0,patch);

    new_dw->getModifiable(mass_src_1,    mi->modelMass_srcLabel,  m1,patch);
    new_dw->getModifiable(momentum_src_1,mi->modelMom_srcLabel,   m1,patch);
    new_dw->getModifiable(energy_src_1,  mi->modelEng_srcLabel,   m1,patch);
    new_dw->getModifiable(sp_vol_src_1,  mi->modelVol_srcLabel,   m1,patch);
    
    new_dw->allocateAndPut(burning,    burningLabel,           m0, patch);  
    new_dw->allocateAndPut(detonating, detonatingLabel,        m0, patch);
    new_dw->allocateAndPut(Fr,         reactedFractionLabel,   m0, patch);
    new_dw->allocateAndPut(delF,       delFLabel,              m0, patch);
    new_dw->allocateAndPut(onSurface,  onSurfaceLabel,         m0, patch);
    new_dw->allocateAndPut(surfaceTemp,surfaceTempLabel,       m0, patch);
    
    new_dw->allocateTemporary(crackedEnough, patch);
    
    Fr.initialize(0.);
    delF.initialize(0.);
    burning.initialize(0.);
    detonating.initialize(0.);
    crackedEnough.initialize(0);
    
    // determing which cells have a crack radius > threshold
    if(d_useCrackModel){
      for(ParticleSubset::iterator iter = pset->begin(); iter != pset->end(); iter++ ){
        IntVector c = level->getCellIndex(px[*iter]);
        
        double crackWidthThreshold = sqrt(8.0e8/pow(press_CC[c],2.84));
        
        if(crackRad[*iter] > crackWidthThreshold) {
          crackedEnough[c] = 1;
        }
      }
    }
    
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m0);
    double cv_rct = mpm_matl->getSpecificHeat();
         
    //__________________________________
    //  Loop over cells
    for (CellIterator iter = patch->getCellIterator();!iter.done();iter++){
      IntVector c = *iter;
      
      // JWL++ Model For explosions
      if (press_CC[c] > d_threshold_press_JWL && rctVolFrac[c] > d_threshold_volFrac){
        
        detonating[c] = 1;   // Flag for detonating 
        
        Fr[c] = prodRho[c]/(rctRho[c]+prodRho[c]);   
        if(Fr[c] >= 0. && Fr[c] < .99){
          delF[c] = d_G*pow(press_CC[c], d_b)*(1.0 - Fr[c]);
        }
        delF[c]*=delT;
        
        double rctMass    = rctRho[c]*cell_vol;
        double prdMass    = prodRho[c]*cell_vol;
        double burnedMass = delF[c]*(prdMass+rctMass);
        
        burnedMass = min(burnedMass, rctMass);
        // 20 % burned mass is a hard limit based p. 55
        //   "JWL++: A Simple Reactive Flow Code Package for Detonation"
        burnedMass = min(burnedMass, .2*mpm_matl->getInitialDensity()*cell_vol);

        totalBurnedMass += burnedMass;

        //__________________________________
        // conservation of mass, momentum and energy                           
        mass_src_0[c] -= burnedMass;
        mass_src_1[c] += burnedMass;         

        Vector momX        = rctvel_CC[c] * burnedMass;
        momentum_src_0[c] -= momX;
        momentum_src_1[c] += momX;
      
        double energyX       = cv_rct * rctTemp[c] * burnedMass; 
        double releasedHeat  = burnedMass * d_E0;
        energy_src_0[c]     -= energyX;
        energy_src_1[c]     += energyX + releasedHeat;
        totalHeatReleased   += releasedHeat;

        double createdVolx  = burnedMass * rctSpvol[c];
        sp_vol_src_0[c]    -= createdVolx;
        sp_vol_src_1[c]    += createdVolx;
        
      } else if(press_CC[c] < d_threshold_press_JWL && press_CC[c] > d_thresholdPress_SB) {
        //__________________________________
        // Find if the cell contains surface:
        IntVector nodeIdx[8];
        patch->findNodesFromCell(*iter,nodeIdx);
        double MaxMass = d_SMALL_NUM;
        double MinMass = 1.0/d_SMALL_NUM; 
                     
        for (int nN=0; nN<8; nN++) {
          double tmp = NC_CCweight[nodeIdx[nN]] * rctMass_NC[nodeIdx[nN]];
          MaxMass = std::max(MaxMass, tmp);
          MinMass = std::min(MinMass, tmp); 
        }               
         
        //===============================================
        //If you change the burning criteria logic you must also modify
        //CCA/Components/SwitchCriteria
        //===============================================
        if ( ((MaxMass-MinMass)/MaxMass > 0.4  &&          //--------------KNOB 1
              (MaxMass-MinMass)/MaxMass < 1.0 &&
               MaxMass > d_TINY_RHO) || 
               crackedEnough[c] ){

          //__________________________________
          //  Determine the temperature
          //  to use in burn model
          double Temp = 0;

          for (int m = 0; m < numAllMatls; m++) {
            if(vol_frac_CC[m][c] > 0.2 && temp_CC[m][c] > d_thresholdTemp && temp_CC[m][c] > Temp ){
              Temp = temp_CC[m][c];
            }
          }  

          surfaceTemp[c] = Temp;

          Vector rhoGradVector = computeDensityGradientVector(nodeIdx, rctMass_NC, NC_CCweight, dx);
          double surfArea = computeSurfaceArea(rhoGradVector, dx);   
          onSurface[c] = surfArea; // debugging var

          //__________________________________
          //  Simple Burn Model
          double burnedMass = 0.0;
          if ((press_CC[c] > d_thresholdPress_SB)) {
            // Flag for burning
            
            if(Temp > d_thresholdTemp){
              burning[c] = 1;
              burnedMass = delT *surfArea * d_BurnCoeff 
                         * pow((press_CC[c]/d_refPress),0.778);
 
              double F = prodRho[c]/(rctRho[c]+prodRho[c]);
              burnedMass += d_Gcrack*(1-F)*pow((press_CC[c]/d_refPress),d_nCrack);
            }

            // Special Temperature criteria for cracking
            if(crackedEnough[c] && Temp < d_thresholdTemp){  
               // if there is appreciable amount of gas above temperature use that temp
               for (int m = 0; m < numAllMatls; m++) {
                  if(vol_frac_CC[m][c] > d_crackVolThreshold && temp_CC[m][c] > 400 && temp_CC[m][c] > Temp ){
                    Temp = temp_CC[m][c];
                  }
                }
                double F = prodRho[c]/(rctRho[c]+prodRho[c]);
                burnedMass += d_Gcrack*(1-F)*pow((press_CC[c]/d_refPress),d_nCrack);

                // std::cout << "Cracked but not regularly burning. Cell: " << c << " Burned Mass: " << burnedMass << " Temperature: " << Temp << endl;
            }


            double rctMass = rctRho[c] * cell_vol;
            if(burnedMass > rctMass){
              burnedMass = rctMass;
            }

            //__________________________________
            // conservation of mass, momentum and energy                           
            mass_src_0[c]   -= burnedMass;
            mass_src_1[c]   += burnedMass;
            totalBurnedMass += burnedMass;

            Vector momX        = rctvel_CC[c] * burnedMass;
            momentum_src_0[c] -= momX;
            momentum_src_1[c] += momX;

            double energyX      = cv_rct*rctTemp[c]*burnedMass; 
            double releasedHeat = burnedMass * d_Enthalpy;
            
            energy_src_0[c]    -= energyX;
            energy_src_1[c]    += energyX + releasedHeat;
            totalHeatReleased  += releasedHeat;

            double createdVolx  = burnedMass * rctSpvol[c];
            sp_vol_src_0[c] -= createdVolx;
            sp_vol_src_1[c] += createdVolx;
          }  // if thresholds have been met
        }  // if (maxMass-MinMass....)
      }  // if (pressure)
    }  // cell iterator  

    //__________________________________
    //  set symetric BC
    setBC(mass_src_0, "set_if_sym_BC",patch, d_sharedState, m0, new_dw);
    setBC(mass_src_1, "set_if_sym_BC",patch, d_sharedState, m1, new_dw);
    setBC(delF,       "set_if_sym_BC",patch, d_sharedState, m0, new_dw);  // I'm not sure you need these???? Todd
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

