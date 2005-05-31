#include <Packages/Uintah/CCA/Components/Models/HEChem/Steady_Burn.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Packages/Uintah/Core/Grid/Variables/NodeIterator.h>
#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Material.h>
#include <Packages/Uintah/Core/Grid/Variables/NCVariable.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCZVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/VarTypes.h>
#include <Packages/Uintah/Core/Labels/MPMLabel.h>
#include <Packages/Uintah/Core/Labels/ICELabel.h>
#include <Packages/Uintah/Core/Labels/MPMICELabel.h>
#include <Packages/Uintah/CCA/Components/ICE/ICEMaterial.h>
#include <Packages/Uintah/CCA/Components/ICE/BoundaryCond.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/CCA/Components/MPM/ParticleCreator/ParticleCreator.h>
#include <Packages/Uintah/Core/Grid/LinearInterpolator.h>

#include <Core/Util/DebugStream.h>

#include <iostream>

using namespace Uintah;
using namespace SCIRun;
using namespace std;
//__________________________________
//  setenv SCI_DEBUG "MPMICE_NORMAL_COUT:+,MPMICE_DOING_COUT:+"
//  MPMICE_DOING_COUT:   dumps when tasks are scheduled and performed
static DebugStream cout_doing("MPMICE_DOING_COUT", false);



Steady_Burn::Steady_Burn(const ProcessorGroup* myworld, ProblemSpecP& params): ModelInterface(myworld), params(params){
  mymatls = 0;
  Ilb = scinew ICELabel();
  MIlb = scinew MPMICELabel();
  //_________________________________
  //  diagnostic labels
  onSurfaceLabel = VarLabel::create("Steady_Burn::onSurface", 
                         CCVariable<double>::getTypeDescription());
  surfaceTempLabel = VarLabel::create("Steady_Burn::surfaceTemp",
                         CCVariable<double>::getTypeDescription());

  PartBulkTempLabel = VarLabel::create("p.BulkTemp",
                         ParticleVariable<double>::getTypeDescription());
  PartBulkTempLabel_preReloc = VarLabel::create("p.BulkTemp+",
                         ParticleVariable<double>::getTypeDescription());
}



Steady_Burn::~Steady_Burn(){
  delete Ilb;
//  delete Mlb;
  delete MIlb;
  
  VarLabel::destroy(surfaceTempLabel);
  VarLabel::destroy(onSurfaceLabel);
  VarLabel::destroy(PartBulkTempLabel);
  VarLabel::destroy(PartBulkTempLabel_preReloc);
  
  if(mymatls && mymatls->removeReference())
    delete mymatls;
}



void Steady_Burn::problemSetup(GridP&, SimulationStateP& sharedState, ModelSetup*){
  cout<<"I am in problem setup" << endl;
  d_sharedState = sharedState;
  matl0 = sharedState->parseAndLookupMaterial(params, "fromMaterial");
  matl1 = sharedState->parseAndLookupMaterial(params, "toMaterial");  
  params->require("PreExpCondPhase",  Ac);
  params->require("ActEnergyCondPh",  Ec);
  params->require("PreExpGasPhase",   Bg);
  params->require("CondPhaseHeat",    Qc);
  params->require("GasPhaseHeat",     Qg);
  params->require("IgnitionTemp",     ignitionTemp);
  
  //__________________________________
  //  define the materialSet
  vector<int> m_tmp(2);
  m_tmp[0] = matl0->getDWIndex();
  m_tmp[1] = matl1->getDWIndex();
  mymatls = new MaterialSet();            

  if( m_tmp[0] != 0 && m_tmp[1] != 0){
    vector<int> m(3);
    m[0] = 0;    // needed for the pressure and NC_CCWeight 
    m[1] = m_tmp[0];
    m[2] = m_tmp[1];
    mymatls->addAll(m);
  }else{
    vector<int> m(2);
    m[0] = m_tmp[0];
    m[1] = m_tmp[1];
    mymatls->addAll(m);
  }
  mymatls->addReference();
}

//______________________________________________________________________
//     
void Steady_Burn::scheduleInitialize(SchedulerP& sched, 
                                     const LevelP& level, const ModelInfo*){
  cout_doing << "Steady_Burn::scheduleInitialize" << endl;
  Task* t = scinew Task("Steady_Burn::initialize", this, &Steady_Burn::initialize);                        
  MaterialSubset* one_matl  = scinew MaterialSubset();
  one_matl->add(0);
  one_matl->addReference();                          
  const MaterialSubset* react_matl = matl0->thisMaterial();
  t->computes(Steady_Burn::surfaceTempLabel,  one_matl);
  t->computes(Steady_Burn::PartBulkTempLabel, react_matl);
  sched->addTask(t, level->eachPatch(), mymatls);
}


//______________________________________________________________________
//  
void
Steady_Burn::initialize(const ProcessorGroup*, 
                        const PatchSubset* patches, 
                        const MaterialSubset* /*matls*/, 
                        DataWarehouse*, 
                        DataWarehouse* new_dw){
  
  int m0 = matl0->getDWIndex();

  (Mlb->d_particleState)[m0].push_back(PartBulkTempLabel);
  (Mlb->d_particleState_preReloc)[m0].push_back(PartBulkTempLabel_preReloc);
  
  for(int p=0;p<patches->size();p++) {
    const Patch* patch = patches->get(p);
    cout_doing << "Doing Initialize on patch " 
               << patch->getID()<< "\t\t\t STEADY_BURN" << endl;         
    CCVariable<double> surfaceTemp;
    new_dw->allocateAndPut(surfaceTemp,Steady_Burn::surfaceTempLabel, 0, patch);
    surfaceTemp.initialize(300.0);

    ParticleVariable<double> pBulkTemp;
    ParticleSubset* pset = new_dw->getParticleSubset(m0, patch);
    new_dw->allocateAndPut(pBulkTemp,  Steady_Burn::PartBulkTempLabel, pset);

    for(ParticleSubset::iterator iter = pset->begin();
                                 iter != pset->end(); iter++){
      particleIndex idx = *iter;
      pBulkTemp[idx]=300.0;
    }
  }
}

//______________________________________________________________________
//      
void Steady_Burn::scheduleComputeStableTimestep(SchedulerP&,
                                                const LevelP&, const ModelInfo*)
{
  // None necessary...
}

//______________________________________________________________________
//     
void Steady_Burn::scheduleComputeModelSources(SchedulerP& sched,
                       const LevelP& level, const ModelInfo* mi){
  Task* t = scinew Task("Steady_Burn::computeModelSources",
                         this, &Steady_Burn::computeModelSources, mi);
  cout_doing << "Steady_Burn::scheduleComputeModelSources" << endl;
  
  t->requires( Task::OldDW, mi->delT_Label);
  Ghost::GhostType  gac = Ghost::AroundCells;  
  Ghost::GhostType  gn  = Ghost::None;
  const MaterialSubset* react_matl = matl0->thisMaterial();
  const MaterialSubset* prod_matl  = matl1->thisMaterial();
  MaterialSubset* one_matl     = scinew MaterialSubset();
  one_matl->add(0);
  one_matl->addReference();
  MaterialSubset* press_matl   = one_matl;

  //__________________________________
  // Products
  t->requires(Task::OldDW,  Ilb->temp_CCLabel,    prod_matl, gn);       
  t->requires(Task::NewDW,  Ilb->vol_frac_CCLabel,prod_matl, gn);       
  t->requires(Task::NewDW,  Ilb->TempX_FCLabel,   prod_matl, gac,2);
  t->requires(Task::NewDW,  Ilb->TempY_FCLabel,   prod_matl, gac,2);
  t->requires(Task::NewDW,  Ilb->TempZ_FCLabel,   prod_matl, gac,2);
  
  //__________________________________
  // Reactants
  t->requires(Task::NewDW, Ilb->sp_vol_CCLabel,   react_matl, gn);
  t->requires(Task::NewDW, Ilb->TempX_FCLabel,    react_matl, gac,2);
  t->requires(Task::NewDW, Ilb->TempY_FCLabel,    react_matl, gac,2);
  t->requires(Task::NewDW, Ilb->TempZ_FCLabel,    react_matl, gac,2);
  t->requires(Task::NewDW, MIlb->vel_CCLabel,     react_matl, gn);
  t->requires(Task::NewDW, MIlb->temp_CCLabel,    react_matl, gn);
  t->requires(Task::NewDW, MIlb->cMassLabel,      react_matl, gn);
  t->requires(Task::NewDW, Mlb->gMassLabel,       react_matl, gac,1);
  t->requires(Task::OldDW, Mlb->pMassLabel,       react_matl, gac,2);
  t->requires(Task::OldDW, PartBulkTempLabel,     react_matl, gac,2);
  t->requires(Task::OldDW, Mlb->pXLabel,          react_matl, gn);
  t->requires(Task::OldDW, Mlb->pSizeLabel,       react_matl, gn);

  //__________________________________
  //Misc
  t->requires(Task::OldDW, surfaceTempLabel,         one_matl, gn);     //new
  t->requires(Task::NewDW, Ilb->press_equil_CCLabel, press_matl,gn);
  t->requires(Task::OldDW, MIlb->NC_CCweightLabel,   one_matl,  gac, 1);    

  t->computes(onSurfaceLabel,              one_matl);
  t->computes(surfaceTempLabel,            one_matl);
  t->computes(PartBulkTempLabel_preReloc,  react_matl);

  t->modifies(mi->mass_source_CCLabel);
  t->modifies(mi->momentum_source_CCLabel);
  t->modifies(mi->energy_source_CCLabel);
  t->modifies(mi->sp_vol_source_CCLabel); 

  sched->addTask(t, level->eachPatch(), mymatls);
  
  if(one_matl->removeReference())
    delete one_matl;
}



void Steady_Burn::scheduleModifyThermoTransportProperties(SchedulerP&,
                                              const LevelP&, const MaterialSet*)
{
  // do nothing      
}



void Steady_Burn::computeSpecificHeat(CCVariable<double>&, const Patch*,
                                      DataWarehouse*, const int)
{
  //do nothing
}
/*
 *
 ***************** Public Member Functions:******************************
 *
 */


//______________________________________________________________________
//
void
Steady_Burn::computeModelSources(const ProcessorGroup*, 
                                 const PatchSubset* patches,
                                 const MaterialSubset* /*matls*/,
                                 DataWarehouse* old_dw,
                                 DataWarehouse* new_dw,
                                 const ModelInfo* mi){
  
  delt_vartype delT;
  old_dw->get(delT, mi->delT_Label);

  //ASSERT(matls->size() == 2);
  int m0 = matl0->getDWIndex();
  int m1 = matl1->getDWIndex();
 
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);  
    
    cout_doing << "Doing massExchange on patch "<< patch->getID()<<"\t\t\t\t Steady_Burn"<<endl;
    CCVariable<double> mass_src_0, mass_src_1, mass_0;
    CCVariable<Vector> momentum_src_0, momentum_src_1;
    CCVariable<double> energy_src_0, energy_src_1;
    CCVariable<double> sp_vol_src_0, sp_vol_src_1;
    CCVariable<double> onSurface, surfaceTemp;   
 
    new_dw->getModifiable(mass_src_0,    mi->mass_source_CCLabel,     m0,patch);
    new_dw->getModifiable(momentum_src_0,mi->momentum_source_CCLabel, m0,patch);
    new_dw->getModifiable(energy_src_0,  mi->energy_source_CCLabel,   m0,patch);
    new_dw->getModifiable(sp_vol_src_0,  mi->sp_vol_source_CCLabel,   m0,patch);

    new_dw->getModifiable(mass_src_1,    mi->mass_source_CCLabel,    m1,patch); 
    new_dw->getModifiable(momentum_src_1,mi->momentum_source_CCLabel,m1,patch);
    new_dw->getModifiable(energy_src_1,  mi->energy_source_CCLabel,  m1,patch);
    new_dw->getModifiable(sp_vol_src_1,  mi->sp_vol_source_CCLabel,  m1,patch);
 
    constCCVariable<double> press_CC,gasTemp,gasVol_frac,gasSp_vol;
    constCCVariable<double> solidTemp,solidMass,solidSp_vol;
    constParticleVariable<double> pMass,pBulkTemp;
    ParticleVariable<double> pBulkTempNew;
    constParticleVariable<Point>  px;
    constParticleVariable<Vector> psize;

    constNCVariable<double> NC_CCweight,NCsolidMass;
    constSFCXVariable<double> gasTempX_FC,solidTempX_FC;
    constSFCYVariable<double> gasTempY_FC,solidTempY_FC;
    constSFCZVariable<double> gasTempZ_FC,solidTempZ_FC;
    constCCVariable<Vector> vel_CC;
    
    Vector dx = patch->dCell();
    Ghost::GhostType  gn  = Ghost::None;    
    Ghost::GhostType  gac = Ghost::AroundCells;
    Ghost::GhostType  gan = Ghost::AroundNodes;

    ParticleSubset* pset = old_dw->getParticleSubset(m0, patch,
                                                     gan, 1, Mlb->pXLabel);
    ParticleSubset* pset_put = old_dw->getParticleSubset(m0, patch);

    //__________________________________
    // Reactant data
    new_dw->get(solidTemp,       MIlb->temp_CCLabel, m0,patch,gn, 0);
    new_dw->get(solidMass,       MIlb->cMassLabel,   m0,patch,gn, 0);
    new_dw->get(solidSp_vol,     Ilb->sp_vol_CCLabel,m0,patch,gn, 0);
    new_dw->get(solidTempX_FC,   Ilb->TempX_FCLabel, m0,patch,gac,2);
    new_dw->get(solidTempY_FC,   Ilb->TempY_FCLabel, m0,patch,gac,2);
    new_dw->get(solidTempZ_FC,   Ilb->TempZ_FCLabel, m0,patch,gac,2);
    new_dw->get(vel_CC,          MIlb->vel_CCLabel,  m0,patch,gn, 0);
    new_dw->get(NCsolidMass,     Mlb->gMassLabel,    m0,patch,gac,1);
    old_dw->get(pMass,           Mlb->pMassLabel,    pset);
    old_dw->get(pBulkTemp,       PartBulkTempLabel,  pset);
    old_dw->get(px,              Mlb->pXLabel,       pset);
    old_dw->get(psize,           Mlb->pSizeLabel,    pset);



    NCVariable<double> NCBulkTemp;
    CCVariable<double> CCBulkTemp;
    new_dw->allocateTemporary(NCBulkTemp, patch, gac, 2);
    new_dw->allocateTemporary(CCBulkTemp, patch);
    NCBulkTemp.initialize(0.0);
    CCBulkTemp.initialize(0.0);
    new_dw->allocateAndPut(pBulkTempNew,  PartBulkTempLabel_preReloc,pset_put);

    //__________________________________
    // Product Data, 
    new_dw->get(gasTempX_FC,      Ilb->TempX_FCLabel,m1,patch,gac,2);
    new_dw->get(gasTempY_FC,      Ilb->TempY_FCLabel,m1,patch,gac,2);
    new_dw->get(gasTempZ_FC,      Ilb->TempZ_FCLabel,m1,patch,gac,2);
    old_dw->get(gasTemp,          Ilb->temp_CCLabel, m1,patch,gn, 0);
    new_dw->get(gasVol_frac,      Ilb->vol_frac_CCLabel,m1,patch,gn,0);

    //________________________________________
    //     Get a new variable from the dw
    new_dw->get(gasSp_vol,        Ilb->sp_vol_CCLabel,m1,patch,gn,0);

    //__________________________________
    //   Misc.
    constCCVariable<double> oldSurfaceTemp; 
    new_dw->get(press_CC,       Ilb->press_equil_CCLabel,      0, patch, gn, 0);
    old_dw->get(NC_CCweight,    MIlb->NC_CCweightLabel,        0, patch, gac,1);
    old_dw->get(oldSurfaceTemp, Steady_Burn::surfaceTempLabel, 0, patch, gn, 0);
    new_dw->allocateAndPut(surfaceTemp,Steady_Burn::surfaceTempLabel, 0, patch);
    new_dw->allocateAndPut(onSurface,  Steady_Burn::onSurfaceLabel,   0, patch);
   
    IntVector nodeIdx[8];

    double Kc=0, Kg=0, Cp=0;
    for (int m = m0; m <=m1; m++) {
      Material* matl = d_sharedState->getMaterial( m );
      ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
      MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
      if(ice_matl){                    // I C E
        Kg = ice_matl->getThermalConductivity();
      }
      if(mpm_matl){                    // M P M
        Kc = mpm_matl->getThermalConductivity();
        Cp = mpm_matl->getSpecificHeat();
      }
    }

    // Interpolate pBulkTemp to NC
    // NOTE!!!  To save myself the headache of passing in the MPMFlags
    // for now I've hardwired this to use linear (8 noded, not GIMP)
    // interpolation.  If you wish to use GIMP, you need to change
    // the interpolator to be a Node27Interpolator() and change
    // n8or27 to be 27 instead of it's current value of 8
    ParticleInterpolator* interpolator;
    interpolator = scinew LinearInterpolator(patch);
    vector<IntVector> ni;
    ni.reserve(interpolator->size());
    vector<double> S;
    S.reserve(interpolator->size());
    int n8or27 = 8;


    NCBulkTemp.initialize(0.);
    for (ParticleSubset::iterator iter = pset->begin();
                                  iter != pset->end(); iter++){
       particleIndex idx = *iter;
       interpolator->findCellAndWeights(px[idx],ni,S,psize[idx]);
        for(int k = 0; k < n8or27; k++) {
          NCBulkTemp[ni[k]]     += pMass[idx]*pBulkTemp[idx]      * S[k];
        }
    }
    for (ParticleSubset::iterator iter = pset_put->begin();
                                  iter != pset_put->end(); iter++){
       particleIndex idx = *iter;
       pBulkTempNew[idx]=pBulkTemp[idx];
    }


    for(NodeIterator iter=patch->getNodeIterator(n8or27);!iter.done();iter++){
        IntVector c = *iter;
        NCBulkTemp[c]/=NCsolidMass[c];
    }

    for(CellIterator iter =patch->getExtraCellIterator();!iter.done();iter++){
      IntVector c = *iter;
      patch->findNodesFromCell(*iter,nodeIdx);
                                                                              
      double Blk_Temp_CC = 0.0;
      double cmass=0.;
      for (int in=0;in<8;in++){
        double NC_CCw_mass = NC_CCweight[nodeIdx[in]]*NCsolidMass[nodeIdx[in]];
        cmass       += NC_CCw_mass;
        Blk_Temp_CC += NCBulkTemp[nodeIdx[in]] * NC_CCw_mass;
      }
      Blk_Temp_CC/=cmass;
      //__________________________________
      // set *_CC = to either vel/Temp_CC_ice or vel/Temp_CC_mpm
      // depending upon if there is cmass.  You need
      // a well defined vel/temp_CC even if there isn't any mass
      // If you change this you must also change
      // MPMICE::computeLagrangianValuesMPM
    }

    for (CellIterator iter = patch->getCellIterator();!iter.done();iter++){
      IntVector c = *iter;
      
      //__________________________________
      // Find if the cell contains surface:
      patch->findNodesFromCell(*iter,nodeIdx);
      double MaxMass = d_SMALL_NUM;
      double MinMass = 1.0/d_SMALL_NUM;
      for (int nN=0; nN<8; nN++){
        MaxMass = std::max(MaxMass,NC_CCweight[nodeIdx[nN]]*NCsolidMass[nodeIdx[nN]]);
        MinMass = std::min(MinMass,NC_CCweight[nodeIdx[nN]]*NCsolidMass[nodeIdx[nN]]); 
      }               
      bool nearSurface =  ( (MaxMass-MinMass)/MaxMass    > 0.4 //--------------KNOB 1
			    && (MaxMass-MinMass)/MaxMass < 1.0
			    &&  MaxMass > d_TINY_RHO ) ? true : false;
      
      //_____________________________________________________________________
      //  Determine if any part of the cell is above the ignition temperature
      double Temp = 0;      
      if (gasVol_frac[c] < 0.2){ //--------------KNOB 2
        Temp =std::max(Temp, solidTempX_FC[c] );    //L
        Temp =std::max(Temp, solidTempY_FC[c] );    //Bot
        Temp =std::max(Temp, solidTempZ_FC[c] );    //BK
        Temp =std::max(Temp, solidTempX_FC[c + IntVector(1,0,0)] );
        Temp =std::max(Temp, solidTempY_FC[c + IntVector(0,1,0)] );
        Temp =std::max(Temp, solidTempZ_FC[c + IntVector(0,0,1)] );
      }else {
        Temp =std::max(Temp, gasTempX_FC[c] );    //L
        Temp =std::max(Temp, gasTempY_FC[c] );    //Bot
        Temp =std::max(Temp, gasTempZ_FC[c] );    //BK
        Temp =std::max(Temp, gasTempX_FC[c + IntVector(1,0,0)] );
        Temp =std::max(Temp, gasTempY_FC[c + IntVector(0,1,0)] );          
        Temp =std::max(Temp, gasTempZ_FC[c + IntVector(0,0,1)] );
      }
      bool hotEnough = (Temp > ignitionTemp) ? true : false;

      if (!(nearSurface && hotEnough)) { // not ignited
        surfaceTemp[c] = solidTemp[c];
        onSurface[c] = 0;
      }
      else {   // cell is ignited
        Vector rhoGradVector = computeDensityGradientVector(nodeIdx, NCsolidMass, NC_CCweight,dx);

	double minSolidTemp = solidTemp[c];
	cout<<"\t To before= "<<minSolidTemp<<endl;

	minSolidTemp = std::min(minSolidTemp, solidTemp[c+IntVector( 1, 1, 1)]);
	minSolidTemp = std::min(minSolidTemp, solidTemp[c+IntVector( 1, 1, 0)]);
	minSolidTemp = std::min(minSolidTemp, solidTemp[c+IntVector( 1, 1,-1)]);
	minSolidTemp = std::min(minSolidTemp, solidTemp[c+IntVector( 0, 1, 1)]);
	minSolidTemp = std::min(minSolidTemp, solidTemp[c+IntVector( 0, 1, 0)]);
	minSolidTemp = std::min(minSolidTemp, solidTemp[c+IntVector( 0, 1,-1)]);
	minSolidTemp = std::min(minSolidTemp, solidTemp[c+IntVector(-1, 1, 1)]);
	minSolidTemp = std::min(minSolidTemp, solidTemp[c+IntVector(-1, 1, 0)]);
	minSolidTemp = std::min(minSolidTemp, solidTemp[c+IntVector(-1, 1,-1)]);

	minSolidTemp = std::min(minSolidTemp, solidTemp[c+IntVector( 1, 0, 1)]);
	minSolidTemp = std::min(minSolidTemp, solidTemp[c+IntVector( 1, 0, 0)]);
	minSolidTemp = std::min(minSolidTemp, solidTemp[c+IntVector( 1, 0,-1)]);
	minSolidTemp = std::min(minSolidTemp, solidTemp[c+IntVector( 0, 0, 1)]);
	minSolidTemp = std::min(minSolidTemp, solidTemp[c+IntVector( 0, 0, 0)]);
	minSolidTemp = std::min(minSolidTemp, solidTemp[c+IntVector( 0, 0,-1)]);
	minSolidTemp = std::min(minSolidTemp, solidTemp[c+IntVector(-1, 0, 1)]);
	minSolidTemp = std::min(minSolidTemp, solidTemp[c+IntVector(-1, 0, 0)]);
	minSolidTemp = std::min(minSolidTemp, solidTemp[c+IntVector(-1, 0,-1)]);

	minSolidTemp = std::min(minSolidTemp, solidTemp[c+IntVector( 1,-1, 1)]);
	minSolidTemp = std::min(minSolidTemp, solidTemp[c+IntVector( 1,-1, 0)]);
	minSolidTemp = std::min(minSolidTemp, solidTemp[c+IntVector( 1,-1,-1)]);
	minSolidTemp = std::min(minSolidTemp, solidTemp[c+IntVector( 0,-1, 1)]);
	minSolidTemp = std::min(minSolidTemp, solidTemp[c+IntVector( 0,-1, 0)]);
	minSolidTemp = std::min(minSolidTemp, solidTemp[c+IntVector( 0,-1,-1)]);
	minSolidTemp = std::min(minSolidTemp, solidTemp[c+IntVector(-1,-1, 1)]);
	minSolidTemp = std::min(minSolidTemp, solidTemp[c+IntVector(-1,-1, 0)]);
	minSolidTemp = std::min(minSolidTemp, solidTemp[c+IntVector(-1,-1,-1)]);


        double surfArea = computeSurfaceArea(rhoGradVector, dx); 
        onSurface[c] = surfArea; // debugging var

        double surfaceTemperature = oldSurfaceTemp[c];
	cout<<"\t Pressure = "<<press_CC[c]<<endl;
	cout<<"\t To after = "<<minSolidTemp<<endl;
	cout<<"\t Rhoc     = "<<1/solidSp_vol[c]<<endl;
	cout<<"\t Rhog     = "<<1/gasSp_vol[c]<<endl;
        double burnedMass = computeBurnedMass(surfaceTemperature, minSolidTemp, Kc, Kg, solidSp_vol[c], gasSp_vol[c], Cp, 
                                              surfArea, delT, solidMass[c]);
	surfaceTemp[c] = surfaceTemperature;
        
	
	//__________________________________
        // conservation of mass, momentum and energy   

	mass_src_0[c] -= burnedMass;
	mass_src_1[c] += burnedMass;
           
        Vector momX = vel_CC[c] * burnedMass;
        momentum_src_0[c] -= momX;
        momentum_src_1[c] += momX;

        double energyX   = Cp*solidTemp[c]*burnedMass; 
        double releasedHeat = burnedMass * (Qc + Qg);
        energy_src_0[c] -= energyX;
        energy_src_1[c] += energyX + releasedHeat;

        double createdVolx = burnedMass * solidSp_vol[c];
        sp_vol_src_0[c] -= createdVolx;
        sp_vol_src_1[c] += createdVolx;
      }  // if (cell is ignited)
    }  // cell iterator  

    //__________________________________
    //  set symetric BC
    setBC(mass_src_0, "set_if_sym_BC",patch, d_sharedState, m0, new_dw);
    setBC(mass_src_1, "set_if_sym_BC",patch, d_sharedState, m1, new_dw); 
    delete interpolator;
  }
}



//______________________________________________________________________

double Steady_Burn::computeSurfaceArea(Vector &rhoGradVector, Vector &dx){
  double delX = dx.x();
  double delY = dx.y();
  double delZ = dx.z();

  double TmpX = fabs(delX*rhoGradVector.x());
  double TmpY = fabs(delY*rhoGradVector.y());
  double TmpZ = fabs(delZ*rhoGradVector.z());
  
  return delX*delY*delZ / (TmpX+TmpY+TmpZ); 
}



Vector Steady_Burn::computeDensityGradientVector(IntVector *nodeIdx, 
						 constNCVariable<double> &NCsolidMass, 
						 constNCVariable<double> &NC_CCweight, 
						 Vector &dx){
  double gradRhoX = 0.25 * (
			    (NCsolidMass[nodeIdx[0]]*NC_CCweight[nodeIdx[0]]+
			     NCsolidMass[nodeIdx[1]]*NC_CCweight[nodeIdx[1]]+
			     NCsolidMass[nodeIdx[2]]*NC_CCweight[nodeIdx[2]]+
			     NCsolidMass[nodeIdx[3]]*NC_CCweight[nodeIdx[3]])
			    -
			    (NCsolidMass[nodeIdx[4]]*NC_CCweight[nodeIdx[4]]+
			     NCsolidMass[nodeIdx[5]]*NC_CCweight[nodeIdx[5]]+
			     NCsolidMass[nodeIdx[6]]*NC_CCweight[nodeIdx[6]]+
			     NCsolidMass[nodeIdx[7]]*NC_CCweight[nodeIdx[7]])
			    )/dx.x();

  double gradRhoY = 0.25 * (
			    (NCsolidMass[nodeIdx[0]]*NC_CCweight[nodeIdx[0]]+
			     NCsolidMass[nodeIdx[1]]*NC_CCweight[nodeIdx[1]]+
			     NCsolidMass[nodeIdx[4]]*NC_CCweight[nodeIdx[4]]+
			     NCsolidMass[nodeIdx[5]]*NC_CCweight[nodeIdx[5]])
			    -
			    (NCsolidMass[nodeIdx[2]]*NC_CCweight[nodeIdx[2]]+
			     NCsolidMass[nodeIdx[3]]*NC_CCweight[nodeIdx[3]]+
			     NCsolidMass[nodeIdx[6]]*NC_CCweight[nodeIdx[6]]+
			     NCsolidMass[nodeIdx[7]]*NC_CCweight[nodeIdx[7]])
			    )/dx.y();

  double gradRhoZ = 0.25 * (
			    (NCsolidMass[nodeIdx[1]]*NC_CCweight[nodeIdx[1]]+
			     NCsolidMass[nodeIdx[3]]*NC_CCweight[nodeIdx[3]]+
			     NCsolidMass[nodeIdx[5]]*NC_CCweight[nodeIdx[5]]+
			     NCsolidMass[nodeIdx[7]]*NC_CCweight[nodeIdx[7]])
			    -
			    (NCsolidMass[nodeIdx[0]]*NC_CCweight[nodeIdx[0]]+
			     NCsolidMass[nodeIdx[2]]*NC_CCweight[nodeIdx[2]]+
			     NCsolidMass[nodeIdx[4]]*NC_CCweight[nodeIdx[4]]+
			     NCsolidMass[nodeIdx[6]]*NC_CCweight[nodeIdx[6]])
			    )/dx.z();

  double absGradRho = sqrt(gradRhoX*gradRhoX + gradRhoY*gradRhoY + gradRhoZ*gradRhoZ );

  return Vector(gradRhoX/absGradRho, gradRhoY/absGradRho, gradRhoZ/absGradRho);
}



//______________________________________________________________________

double Steady_Burn::computeBurnedMass(double& Ts, double To, double Kc, double Kg, double Vc, double Vg, double Cp,
				      double surfArea, double delT, double solidMass){
  //Minimum Surface Temperature
  double Tmin = To +Qc/Cp; 
    //(Qc - Ec*Cp + 2*To*Cp + sqrt(Qc*Qc+4*Qc*To*Cp+Ec*Ec*Cp*Cp+4*To*To*Cp*Cp))/(2*Cp);
  
  if (Ts < Tmin) Ts = Tmin;
  
  Ts = computeWSBSurfaceTemp(Ts, To, Kc, Kg, Vc, Vg, Cp);    
  double Mr = computeMassTransferRate(Ts, To, Kc, Vc, Cp);
  cout<<"\t Ts = "<<Ts<<endl;
  // cout<<"\t Mr = "<<Mr<<endl;
  cout<<"\t r  = "<<Mr*Vc<<endl;
  double burnedMass = delT * surfArea * Mr;
  if (burnedMass > solidMass) 
    burnedMass = solidMass;  
  return burnedMass;
}



//_________________________________________________________
double Steady_Burn::computeWSBSurfaceTemp(double Ts, double To, double Kc, double Kg, double Vc, double Vg, double Cp){
  double Mr = 0.0;
  double Xg = 0.0;
  double diff = 1.0;
  int iter = 0;
  int n = 3;   // used to compute n-point running average of T_s
  double Tsnew = Ts;    // temporary storage for surfaceTemp
  
  while (diff > 1.0E-6) {
    Mr = computeMassTransferRate(Ts, To, Kc, Vc, Cp);
    Xg = 2.0/(sqrt(Mr*Mr+4*Kg*Bg*Ts*Ts/(Vg*Vg*Cp)) - Mr);    
    Tsnew = (Tsnew*(n-1) + To + Qc/Cp + Qg/(Cp*(Xg*Mr+1)))/n;
    diff = fabs((Tsnew - Ts)/Tsnew);
    Ts = Tsnew;
    iter++;
    
    if (iter > 10){
      n++;
      iter = 0;
    }
  }
  
  return Ts;
}



//______________________________________________________________________
double Steady_Burn::computeMassTransferRate(double Ts, double To, double Kc, double Vc, double Cp){
  double Tmin = To +Qc/Cp; 
  if (Ts < Tmin) Ts = Tmin;
  
  double Mr = sqrt(Ac*Kc*Ts*Ts*exp(-Ec/Ts)/(Vc*Ec*((Ts-To)*Cp-Qc/2)));
  return Mr;
}

//______________________________________________________________________
//
void Steady_Burn::scheduleErrorEstimate(const LevelP&,
                                         SchedulerP&)
{
  // Not implemented yet
}
//__________________________________
void Steady_Burn::scheduleTestConservation(SchedulerP&,
                                           const PatchSet*,                      
                                           const ModelInfo*)                     
{
  // Not implemented yet
}

void Steady_Burn::setMPMLabel(MPMLabel* MLB)
{
  Mlb = MLB;
}
