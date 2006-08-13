#include <Packages/Uintah/CCA/Components/Models/HEChem/Unsteady_Burn.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/Variables/VarTypes.h>
#include <Packages/Uintah/Core/Labels/MPMLabel.h>
#include <Packages/Uintah/Core/Labels/ICELabel.h>
#include <Packages/Uintah/Core/Labels/MPMICELabel.h>
#include <Packages/Uintah/CCA/Components/ICE/ICEMaterial.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/CCA/Components/ICE/BoundaryCond.h>
#include <Packages/Uintah/Core/Exceptions/InvalidValue.h>
#include <Core/Util/DebugStream.h>

#include <iostream>

#undef TOTALS
//#define TOTALS    


using namespace Uintah;
using namespace SCIRun;
using namespace std;
//__________________________________   
//  setenv SCI_DEBUG "MPMICE_NORMAL_COUT:+,MPMICE_DOING_COUT:+"
//  MPMICE_DOING_COUT:   dumps when tasks are scheduled and performed
static DebugStream cout_doing("MPMICE_DOING_COUT", false);

const double Unsteady_Burn::INIT_TS = 300.0;    /* initial surface temperature          */
const double Unsteady_Burn::INIT_BETA = 1.0e12; /* initial surface temperature gradient */
const double Unsteady_Burn::EPS = 1.e-5;        /* iteration stopping criterion */
const double Unsteady_Burn::UNDEFINED = -10; 

Unsteady_Burn::Unsteady_Burn(const ProcessorGroup* myworld, ProblemSpecP& params)
  : ModelInterface(myworld), params(params) { 
  mymatls = 0;
  Mlb  = scinew MPMLabel();
  Ilb  = scinew ICELabel();
  MIlb = scinew MPMICELabel();

  BurningCellLabel  = VarLabel::create("UnsteadyBurn.BurningCell",            CCVariable<double>::getTypeDescription());
  TsLabel           = VarLabel::create("UnsteadyBurn.SurfTemp",               CCVariable<double>::getTypeDescription());
  BetaLabel         = VarLabel::create("UnsteadyBurn.SurfTempGrad",           CCVariable<double>::getTypeDescription());
  PartBetaLabel     = VarLabel::create("UnsteadyBurn.PartSurfTempGrad", ParticleVariable<double>::getTypeDescription());
  PartTsLabel       = VarLabel::create("UnsteadyBurn.PartSurfTemp",     ParticleVariable<double>::getTypeDescription());
}


Unsteady_Burn::~Unsteady_Burn(){
  delete Ilb;
  //delete Mlb; /* don't delete it here, or complain "double free or corruption" */
  delete MIlb;

  VarLabel::destroy(BurningCellLabel);
  VarLabel::destroy(TsLabel);
  VarLabel::destroy(BetaLabel);
  VarLabel::destroy(PartBetaLabel);
  VarLabel::destroy(PartTsLabel);

  if(mymatls && mymatls->removeReference())
    delete mymatls;
}


void Unsteady_Burn::problemSetup(GridP&, SimulationStateP& sharedState, ModelSetup*){
  cout<<"I am in problem setup" << endl;
  d_sharedState = sharedState;
  matl0 = sharedState->parseAndLookupMaterial(params, "fromMaterial");
  matl1 = sharedState->parseAndLookupMaterial(params, "toMaterial");  

  params->require("IdealGasConst",     R );
  params->require("PreExpCondPh",      Ac);
  params->require("ActEnergyCondPh",   Ec);
  params->require("PreExpGasPh",       Bg);
  params->require("CondPhaseHeat",     Qc);
  params->require("GasPhaseHeat",      Qg);
  params->require("HeatConductGasPh",  Kg);
  params->require("HeatConductCondPh", Kc);
  params->require("SpecificHeatBoth",  Cp);
  params->require("MoleWeightGasPh",   MW);
  params->require("BoundaryParticles", BP);
  params->require("BurnrateModCoef",   Bm);
  params->require("CondUnsteadyCoef",  Nc);
  params->require("GasUnsteadyCoef",   Ng);
  params->require("ThresholdPressure", ThresholdPressure);
  params->require("IgnitionTemp",      ignitionTemp);

  /* initialize constants */
  CC1 = Ac*R*Kc/Ec/Cp;        
  CC2 = Qc/Cp/2;              
  CC3 = 4*Kg*Bg*MW*MW/Cp/R/R;  
  CC4 = Qc/Cp;                
  CC5 = Qg/Cp;                
  
  /*  define the materialSet */
  vector<int> m_tmp(2);
  m_tmp[0] = matl0->getDWIndex();
  m_tmp[1] = matl1->getDWIndex();
  mymatls  = new MaterialSet();            
  
  if( m_tmp[0] != 0 && m_tmp[1] != 0){
    vector<int> m(3);
    m[0] = 0; /* needed for the pressure and NC_CCWeight */ 
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


void Unsteady_Burn::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP model_ps = ps->appendChild("Model");
  model_ps->setAttribute("type","Unsteady_Burn");

  model_ps->appendElement("fromMaterial",      matl0->getName());
  model_ps->appendElement("toMaterial",        matl1->getName());

  model_ps->appendElement("IdealGasConst",     R );
  model_ps->appendElement("PreExpCondPh",      Ac);
  model_ps->appendElement("ActEnergyCondPh",   Ec);
  model_ps->appendElement("PreExpGasPh",       Bg);
  model_ps->appendElement("CondPhaseHeat",     Qc);
  model_ps->appendElement("GasPhaseHeat",      Qg);
  model_ps->appendElement("HeatConductGasPh",  Kg);
  model_ps->appendElement("HeatConductCondPh", Kc);
  model_ps->appendElement("SpecificHeatBoth",  Cp);
  model_ps->appendElement("MoleWeightGasPh",   MW);
  model_ps->appendElement("BoundaryParticles", BP);
  model_ps->appendElement("BurnrateModCoef",   Bm);
  model_ps->appendElement("CondUnsteadyCoef",  Nc);
  model_ps->appendElement("GasUnsteadyCoef",   Ng);
  model_ps->appendElement("ThresholdPressure", ThresholdPressure);
  model_ps->appendElement("IgnitionTemp",      ignitionTemp);
}


void Unsteady_Burn::scheduleInitialize(SchedulerP& sched, const LevelP& level, const ModelInfo*){
  cout_doing << "Unsteady_Burn::scheduleInitialize" << endl;
  Task* t = scinew Task("Unsteady_Burn::initialize", this, &Unsteady_Burn::initialize);                        
  const MaterialSubset* react_matl = matl0->thisMaterial();
  t->computes(BurningCellLabel, react_matl);
  t->computes(TsLabel,          react_matl);
  t->computes(BetaLabel,        react_matl);
  t->computes(PartBetaLabel,    react_matl);
  t->computes(PartTsLabel,      react_matl);
  sched->addTask(t, level->eachPatch(), mymatls);
}


void Unsteady_Burn::initialize(const ProcessorGroup*, 
			     const PatchSubset* patches, 
			     const MaterialSubset* /*matls*/, 
			     DataWarehouse*, 
			     DataWarehouse* new_dw){
  int m0 = matl0->getDWIndex();
  for(int p=0;p<patches->size();p++) {
    const Patch* patch = patches->get(p);
    cout_doing << "Doing Initialize on patch " << patch->getID()<< "\t\t\t UNSTEADY_BURN" << endl; 

    CCVariable<double> BurningCell;
    new_dw->allocateAndPut(BurningCell, BurningCellLabel, m0, patch);
    BurningCell.initialize(0.0);

    CCVariable<double> Ts;
    new_dw->allocateAndPut(Ts, TsLabel, m0, patch);
    Ts.initialize(INIT_TS);

    CCVariable<double> Beta;
    new_dw->allocateAndPut(Beta, BetaLabel, m0, patch);
    Beta.initialize(INIT_BETA);

    ParticleVariable<double> pBeta, pTs;
    ParticleSubset* pset_gn = new_dw->getParticleSubset(m0, patch);
    new_dw->allocateAndPut(pBeta, PartBetaLabel, pset_gn);
    new_dw->allocateAndPut(pTs,   PartTsLabel,   pset_gn);

    for(ParticleSubset::iterator iter=pset_gn->begin(), iter_end=pset_gn->end(); iter != iter_end; iter++){
      particleIndex idx = *iter;
      pBeta[idx] = INIT_BETA;
      pTs[idx]   = INIT_TS;
    }
  }        
}


void Unsteady_Burn::scheduleComputeStableTimestep(SchedulerP&, const LevelP&, const ModelInfo*){
  // None necessary...
}


void Unsteady_Burn::scheduleComputeModelSources(SchedulerP& sched, const LevelP& level, const ModelInfo* mi){
  Task* t = scinew Task("Unsteady_Burn::computeModelSources", this, &Unsteady_Burn::computeModelSources, mi);
  cout_doing << "Unsteady_Burn::scheduleComputeModelSources" << endl;
  
  t->requires( Task::OldDW, mi->delT_Label);

  Ghost::GhostType gac = Ghost::AroundCells;  
  Ghost::GhostType gn  = Ghost::None;

  const MaterialSubset* react_matl = matl0->thisMaterial();
  MaterialSubset* one_matl   = scinew MaterialSubset();
  one_matl->add(0);
  one_matl->addReference();

  /*
    const MaterialSubset* ice_matls = d_sharedState->allICEMaterials()->getUnion();
    const MaterialSubset* mpm_matls = d_sharedState->allMPMMaterials()->getUnion();
  */
  
  t->requires(Task::OldDW, Ilb->temp_CCLabel,     gac,1);
  t->requires(Task::NewDW, Ilb->vol_frac_CCLabel, gac,1);
  /*     Products     */
  /*     Reactants    */
  t->requires(Task::NewDW, Ilb->sp_vol_CCLabel,    react_matl, gn);
  t->requires(Task::NewDW, MIlb->vel_CCLabel,      react_matl, gn);
  t->requires(Task::NewDW, MIlb->cMassLabel,       react_matl, gn);
  t->requires(Task::NewDW, MIlb->gMassLabel,       react_matl, gac,1);
  t->requires(Task::OldDW, Mlb->pXLabel,           react_matl, gac,1);
  /*     Misc      */
  t->requires(Task::NewDW, Ilb->press_equil_CCLabel, one_matl, gac, 1);
  t->requires(Task::OldDW, MIlb->NC_CCweightLabel,   one_matl, gac, 1);  

  t->requires(Task::OldDW, BurningCellLabel,       react_matl, gac, 1);     
  t->requires(Task::OldDW, TsLabel,                react_matl, gac, 1);     
  t->requires(Task::OldDW, BetaLabel,              react_matl, gac, 1);     
  t->requires(Task::OldDW, PartBetaLabel,          react_matl, gn);
  t->requires(Task::OldDW, PartTsLabel,            react_matl, gn);

  t->computes(BurningCellLabel, react_matl);  
  t->computes(TsLabel,          react_matl);
  t->computes(BetaLabel,        react_matl);
  t->computes(PartBetaLabel,    react_matl);
  t->computes(PartTsLabel,      react_matl);
   
  t->modifies(mi->mass_source_CCLabel);
  t->modifies(mi->momentum_source_CCLabel);
  t->modifies(mi->energy_source_CCLabel);
  t->modifies(mi->sp_vol_source_CCLabel); 

  sched->addTask(t, level->eachPatch(), mymatls);
  if(one_matl->removeReference())
    delete one_matl;
}


void Unsteady_Burn::scheduleModifyThermoTransportProperties(SchedulerP&, const LevelP&, const MaterialSet*){
  // do nothing      
}


void Unsteady_Burn::computeSpecificHeat(CCVariable<double>&, const Patch*, DataWarehouse*, const int){
  //do nothing
}



/*
 ***************** Private Member Functions:******************************
 */
void Unsteady_Burn::computeModelSources(const ProcessorGroup*, 
				      const PatchSubset* patches,
				      const MaterialSubset* /*matls*/,
				      DataWarehouse* old_dw,
				      DataWarehouse* new_dw,
				      const ModelInfo* mi){
  
  delt_vartype delT;
  old_dw->get(delT, mi->delT_Label);
  
  //ASSERT(matls->size() == 2);
  int m0 = matl0->getDWIndex(); /* reactant material */
  int m1 = matl1->getDWIndex(); /* product material */

  Ghost::GhostType gn  = Ghost::None;    
  Ghost::GhostType gac = Ghost::AroundCells;  

  /* Patch Iteration */
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);  
    
    cout_doing << "Doing massExchange on patch "<< patch->getID()<<"\t\t\t\t Unsteady_Burn"<<endl;
    CCVariable<double> mass_src_0, mass_src_1, mass_0;
    CCVariable<Vector> momentum_src_0, momentum_src_1;
    CCVariable<double> energy_src_0, energy_src_1;
    CCVariable<double> sp_vol_src_0, sp_vol_src_1;
    /* reactant */
    new_dw->getModifiable(mass_src_0,     mi->mass_source_CCLabel,     m0, patch);    
    new_dw->getModifiable(momentum_src_0, mi->momentum_source_CCLabel, m0, patch); 
    new_dw->getModifiable(energy_src_0,   mi->energy_source_CCLabel,   m0, patch);   
    new_dw->getModifiable(sp_vol_src_0,   mi->sp_vol_source_CCLabel,   m0, patch);   
    /* product */
    new_dw->getModifiable(mass_src_1,     mi->mass_source_CCLabel,     m1, patch);   
    new_dw->getModifiable(momentum_src_1, mi->momentum_source_CCLabel, m1, patch);   
    new_dw->getModifiable(energy_src_1,   mi->energy_source_CCLabel,   m1, patch);   
    new_dw->getModifiable(sp_vol_src_1,   mi->sp_vol_source_CCLabel,   m1, patch);   
    
    constCCVariable<double>   press_CC, solidTemp, solidMass, solidSp_vol;
    constNCVariable<double>   NC_CCweight, NCsolidMass;
    constCCVariable<Vector>   vel_CC;

    constCCVariable<double>  OldBurningCell, OldTs, OldBeta;
    CCVariable<double>       NewBurningCell, NewTs, NewBeta;
    constParticleVariable<double> pBetaOld, pTsOld;
    ParticleVariable<double>      pBetaNew, pTsNew;
    
    /* Reactant data */
    old_dw->get(OldBurningCell, BurningCellLabel, m0, patch, gac, 1);
    old_dw->get(OldTs,                   TsLabel, m0, patch, gac, 1);
    old_dw->get(OldBeta,               BetaLabel, m0, patch, gac, 1);

    old_dw->get(solidTemp,       MIlb->temp_CCLabel,    m0, patch, gac, 1);
    new_dw->get(solidMass,       MIlb->cMassLabel,      m0, patch, gn,  0);
    new_dw->get(solidSp_vol,     Ilb->sp_vol_CCLabel,   m0, patch, gn,  0);   
    new_dw->get(vel_CC,          MIlb->vel_CCLabel,     m0, patch, gn,  0);
    new_dw->get(NCsolidMass,     MIlb->gMassLabel,      m0, patch, gac, 1);
 
    /* for burning surface definition (BoundaryParticles) */
    constParticleVariable<Point>  px_gac;    
    ParticleSubset* pset_gac = old_dw->getParticleSubset(m0, patch, gac, 1, Mlb->pXLabel);
    old_dw->get(px_gac, Mlb->pXLabel, pset_gac);

    /* for unsteay burn parameters stored on particles */
    constParticleVariable<Point>  px_gn; 
    ParticleSubset* pset_gn = old_dw->getParticleSubset(m0, patch);
    old_dw->get(px_gn, Mlb->pXLabel, pset_gn);
  
    /* Indicating cells containing how many particles */
    CCVariable<double> pFlag;
    new_dw->allocateTemporary(pFlag, patch, gac, 1);
    pFlag.initialize(0.0);
    
    /* Product Data */
    /* Misc */
    new_dw->get(press_CC,    Ilb->press_equil_CCLabel, 0, patch, gac, 1);
    old_dw->get(NC_CCweight, MIlb->NC_CCweightLabel,   0, patch, gac, 1);

    new_dw->allocateAndPut(NewBurningCell, BurningCellLabel, m0, patch, gac, 1);
    new_dw->allocateAndPut(NewTs,          TsLabel,          m0, patch, gac, 1);
    new_dw->allocateAndPut(NewBeta,        BetaLabel,        m0, patch, gac, 1);

    old_dw->get           (pBetaOld, PartBetaLabel, pset_gn);
    old_dw->get           (pTsOld,   PartTsLabel,   pset_gn);
    new_dw->allocateAndPut(pBetaNew, PartBetaLabel, pset_gn);
    new_dw->allocateAndPut(pTsNew,   PartTsLabel,   pset_gn);
		
    /* All Material Data */
    int numAllMatls = d_sharedState->getNumMatls();
    StaticArray<constCCVariable<double> >  vol_frac_CC(numAllMatls);
    StaticArray<constCCVariable<double> >  temp_CC(numAllMatls);
    for(int m = 0; m < numAllMatls; m++){
      Material* matl = d_sharedState->getMaterial(m);
      int indx = matl->getDWIndex();
      old_dw->get(temp_CC[m],     MIlb->temp_CCLabel,    indx, patch, gac, 1);
      new_dw->get(vol_frac_CC[m], Ilb->vol_frac_CCLabel, indx, patch, gac, 1);
    }
 
#ifdef TOTALS
    double totalSurfArea=0.0, totalBurnedMass=0.0;
    int totalNumBurningCells=0;
#endif      

   /* initialize NEW cell-centered and particle values for unsteady burn */
    NewBurningCell.initialize(0.0);
    NewTs.initialize(0.0);
    NewBeta.initialize(0.0);
    for(ParticleSubset::iterator iter=pset_gn->begin(), iter_end=pset_gn->end(); iter != iter_end; iter++){
       particleIndex idx = *iter;
       pBetaNew[idx] = INIT_BETA;
       pTsNew[idx]   = INIT_TS;
    }

    /* count how many reactant particles in each cell for burn surface definition */
    for(ParticleSubset::iterator iter=pset_gac->begin(), iter_end=pset_gac->end(); iter != iter_end; iter++){
      particleIndex idx = *iter;
      IntVector c;
      patch->findCell(px_gac[idx],c);
      pFlag[c] += 1.0;
    }
    setBC(pFlag, "set_if_sym_BC", patch, d_sharedState, m0, new_dw);

    /* Initialize Cell-centered Ts and Beta with OLD particle centered beta value, 
       The CC value takes the largest particle beta value in the cell that
       must be less than the INIT values   */
    for(ParticleSubset::iterator iter=pset_gn->begin(), iter_end=pset_gn->end(); iter != iter_end; iter++){
      particleIndex idx = *iter;
      IntVector c;
      patch->findCell(px_gn[idx],c);
      if(pBetaOld[idx]<INIT_BETA && pBetaOld[idx]>NewBeta[c]){
	NewBeta[c] = pBetaOld[idx];  
	NewTs[c]   = pTsOld[idx];
      }
    }
    
    /* Cell Iteration */
    Vector dx = patch->dCell(); 
    IntVector nodeIdx[8];
    for (CellIterator iter = patch->getCellIterator();!iter.done();iter++){
      IntVector c = *iter;
      patch->findNodesFromCell(*iter,nodeIdx);

      double MaxMass = d_SMALL_NUM;
      double MinMass = 1.0/d_SMALL_NUM; 
      for (int nN=0; nN<8; nN++){
	MaxMass = std::max(MaxMass,NC_CCweight[nodeIdx[nN]]*NCsolidMass[nodeIdx[nN]]);
	MinMass = std::min(MinMass,NC_CCweight[nodeIdx[nN]]*NCsolidMass[nodeIdx[nN]]); 
      }

      /* test whether the current cell satisfies burning criteria */
      bool   burning = 0;
      double maxProductVolFrac  = -1.0;
      double maxReactantVolFrac = -1.0;
      double productPress = 0.0;
      double Tzero = 0.0;
      double temp_vf = 0.0;
      /*if( (MaxMass-MinMass)/MaxMass>0.4 && (MaxMass-MinMass)/MaxMass<1.0 && pFlag[c]>0 ){ */
      if( MinMass/MaxMass<0.7 && pFlag[c]>0 ){ 
	/* near interface and containing particles */
	for(int i = -1; i<=1; i++){
	  for(int j = -1; j<=1; j++){
	    for(int k = -1; k<=1; k++){
	      IntVector cell = c + IntVector(i,j,k);

	      /* Search for Tzero from max_vol_frac reactant cell */
	      temp_vf = vol_frac_CC[m0][cell]; 
	      if( temp_vf > maxReactantVolFrac ){
		maxReactantVolFrac = temp_vf;
		Tzero = solidTemp[cell];
	      }//endif

	      /* Search for pressure from max_vol_frac product cell */
	      temp_vf = vol_frac_CC[m1][cell]; 
	      if( temp_vf > maxProductVolFrac ){
		maxProductVolFrac = temp_vf;
		productPress = press_CC[cell];
	      }//endif
	      
	      if(burning == 0 && pFlag[cell] <= BP){
		for (int m = 0; m < numAllMatls; m++){
		  if(vol_frac_CC[m][cell] > 0.3 && temp_CC[m][cell] > ignitionTemp)   
		    burning = 1;
		}
	      }//endif

	    }//end 1st for
	  }//end 2nd for
	}//end 3rd for
      }//endif
      
      if(burning == 1 && productPress >= ThresholdPressure){
	Vector rhoGradVector = computeDensityGradientVector(nodeIdx, NCsolidMass, NC_CCweight,dx);
       	double surfArea = computeSurfaceArea(rhoGradVector, dx); 
	
	/* If particles in a cell are newly ignited, their initial values 
	   (Ts and Beta) are copied from a neighboring cell that has been 
	   burning the longest time. Or, if no such cell, then they take the 
	   INIT values	 */ 
	IntVector mostBurntCell;
	double maxBurning = 0.0;
	/* if the cell did not burn in the last timestep */
	if(NewBeta[c] == 0.0 || NewTs[c] == 0.0){
	  /* find cell has been burnt the longest */ 
	  for(int i = -1; i<=1; i++){
	    for(int j = -1; j<=1; j++){
	      for(int k = -1; k<=1; k++){
		// loop through all neighboring cells 
		IntVector cell = c + IntVector(i,j,k);
		if(OldBurningCell[cell]>maxBurning){
		  maxBurning = OldBurningCell[cell];
		  mostBurntCell = cell;
		}		
	      }//end 3rd for
	    }//end 2nd for
	  }//end 1st for
	  
	  if(maxBurning > 0.0){
	    NewBeta[c] = OldBeta[mostBurntCell];
	    NewTs[c]   =   OldTs[mostBurntCell];    
	  }else{
	    NewBeta[c] = INIT_BETA;
	    NewTs[c]   = INIT_TS;
	  }
	}

	double beta = NewBeta[c];
	double Ts   = NewTs[c]; 

	double burnedMass = computeBurnedMass(Tzero, productPress, solidSp_vol[c], surfArea,
					      delT, solidMass[c], beta, Ts, dx);
	
	NewBeta[c] = beta;
	NewTs[c]   = Ts;

	NewBurningCell[c] = OldBurningCell[c] + 1.0;	

#ifdef TOTALS
	totalSurfArea += surfArea;
	totalBurnedMass += burnedMass;
	totalNumBurningCells++;
#endif
	
	/* conservation of mass, momentum and energy   */
	mass_src_0[c]  -= burnedMass;
	mass_src_1[c]    += burnedMass;
	
        Vector momX = vel_CC[c] * burnedMass;
        momentum_src_0[c]  -= momX;
        momentum_src_1[c]    += momX;
	
        double energyX   = Cp*solidTemp[c]*burnedMass; 
        double releasedHeat = burnedMass * (Qc + Qg);
        energy_src_0[c]  -= energyX;
        energy_src_1[c]    += energyX + releasedHeat;

        double createdVolx = burnedMass * solidSp_vol[c];
        sp_vol_src_0[c]  -= createdVolx;
        sp_vol_src_1[c]    += createdVolx;
      }/* end if (cell is ignited) */
    }/* End Cell Iteration */

    /* update newly calculated CC values to particles */
    for(ParticleSubset::iterator iter=pset_gn->begin(), iter_end=pset_gn->end(); iter != iter_end; iter++){
      particleIndex idx = *iter;
      IntVector c;
      patch->findCell(px_gn[idx],c);
      if(NewBurningCell[c]>0.0){
	pBetaNew[idx] = NewBeta[c];
	pTsNew[idx]   = NewTs[c];
      }else{
	pBetaNew[idx] = INIT_BETA;
	pTsNew[idx]   = INIT_TS;
      }
    }
    
#ifdef TOTALS    
    if(0.0 != totalBurnedMass){
      cout<<"  TotalSurfArea = "<<totalSurfArea <<"   TotalBurnedMass = " <<totalBurnedMass 
	  <<"   TotalNumBurningCells = "<<totalNumBurningCells<<endl;
    }
#endif

    /*  set symetric BC  */
    setBC(mass_src_0, "set_if_sym_BC", patch, d_sharedState, m0, new_dw);
    setBC(mass_src_1, "set_if_sym_BC", patch, d_sharedState, m1, new_dw);

    setBC(NewBurningCell, "set_if_sym_BC", patch, d_sharedState, m0, new_dw);
    setBC(NewTs,          "set_if_sym_BC", patch, d_sharedState, m0, new_dw);
    setBC(NewBeta,        "set_if_sym_BC", patch, d_sharedState, m0, new_dw); 
  }
}

//______________________________________________________________________

double Unsteady_Burn::computeSurfaceArea(Vector &rhoGradVector, Vector &dx){
  double delX = dx.x();
  double delY = dx.y();
  double delZ = dx.z();
  double rgvX = fabs(rhoGradVector.x());
  double rgvY = fabs(rhoGradVector.y());
  double rgvZ = fabs(rhoGradVector.z());
    
  double max = rgvX;
  if(rgvY > max)   max = rgvY;
  if(rgvZ > max)   max = rgvZ;
  
  double coeff = pow(1.0/max, 1.0/3.0);
   
  double TmpX = delX*rgvX;
  double TmpY = delY*rgvY;
  double TmpZ = delZ*rgvZ;
    
  return delX*delY*delZ / (TmpX+TmpY+TmpZ) * coeff; 
}


Vector Unsteady_Burn::computeDensityGradientVector(IntVector *nodeIdx, 
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

void Unsteady_Burn::scheduleErrorEstimate(const LevelP&, SchedulerP&){
  // Not implemented yet
}

void Unsteady_Burn::scheduleTestConservation(SchedulerP&, const PatchSet*, const ModelInfo*){
  // Not implemented yet
}

void Unsteady_Burn::setMPMLabel(MPMLabel* MLB){
  Mlb = MLB;
}


/****************************************************************************/
/******************* Bisection Secant Solver ********************************/
/****************************************************************************/
double Unsteady_Burn::computeBurnedMass(double To, double P, double Vc, double surfArea, double delT, 
					double solidMass, double& beta, double& Ts, Vector& dx){  
  UpdateConstants(To, P, Vc);

  double   Ts_local = Tmin + (Tmax - Tmin) * BisectionSecant();
  double    m_local = sqrt(C1*Ts_local*Ts_local/(Ts_local-C2)*exp(-Ec/(R*Ts_local)));
  double beta_local = (Ts_local-To) * m_local * K1;
  double m_unsteady = 0.0;

  if(beta<INIT_BETA && Ts>INIT_TS){
    double b = K3*beta;
    double c = K4*Ts*Ts*exp(-Ec/(R*Ts));
    double para = b*b-4*c;
    if(para < 0){
      throw InvalidValue("Timestep is too large to correctly compute the unsteady burn rate", __FILE__, __LINE__);
    }
    m_unsteady = 2*c/(sqrt(para)+b);
 
    /* manually adjust the unsteady burn rate based on the local steady burn rate */
    m_unsteady = m_local * pow(m_unsteady/m_local, Bm);

    double n1 = (K2/(delT*m_unsteady*m_unsteady)) * Nc;
    double n2 = n1 * Ng;
    if(n1>1)
      beta = (n1-1)/n1*beta + 1/n1*beta_local;
    else 
      beta = beta_local;
    
    if(n2>1)
      Ts   = (n2-1)/n2*Ts   + 1/n2*Ts_local;
    else 
      Ts = Ts_local;

    //cout <<"  beta="<< beta <<" Ts="<< Ts <<" n1="<< n1<<endl;
  
  }else{
    beta = beta_local;
    Ts = Ts_local;
    m_unsteady = m_local;
  }
   
  double burnedMass = delT * surfArea * m_unsteady;
  double min_mass = dx.x()*dx.y()*dx.z()*d_TINY_RHO;
  if (burnedMass + min_mass > solidMass) 
    burnedMass = solidMass - min_mass;  
  return burnedMass;
}


void Unsteady_Burn::UpdateConstants(double To, double P, double Vc){
  /* define the constants as:   */
  /* CC1 = Ac*R*Kc/Ec/Cp        */
  /* CC2 = Qc/Cp/2              */
  /* CC3 = 4*Kg*Bg*W*W/Cp/R/R;  */
  /* CC4 = Qc/Cp                */
  /* CC5 = Qg/Cp                */
  C1 = CC1 / Vc; 
  /* Vc = Condensed Phase Specific Volume */
  C2 = To + CC2; 
  C3 = CC3 * P*P;
  C4 = To + CC4; 
  C5 = CC5 * C3; 

  K1 = Cp/Kc;                 
  K2 = Kc/(Cp*Vc);            
  K3 = 2*Kc/Qc;              
  K4 = 2*Ac*R*Kc/(Ec*Qc*Vc);

  L0 = 0; R0 = 1;
  L1 = 0; R1 = 1;
  L2 = 0; R2 = 1;
  L3 = 0; R3 = 1;
  
  T_ignition = C2;
  Tmin = Ts_max();
  Tmax = Fxn_Ts(Tmin);
}


/***   
 ***   Ts = f_Ts(Ts)						   
 ***   f_Ts(Ts) = C4 + C5/(sqrt(m^2+C3) + m)^2 
 ***
 ***   Solve for diff(f_Ts(Ts))=0 ***   Ts_max = C2 - Ec/2R + sqrt(4*R^2*C2^2+Ec^2)/2R
 ***   f_Ts_max = f_Ts(Ts_max)
 ***/
double Unsteady_Burn::Fxn_Ts(double Ts){
  double m2 = C1*Ts*Ts/(Ts-C2)*exp(-Ec/R/Ts);
  double m = sqrt(m2);
  //double result = C4 + C5/pow((sqrt(m2+C3)+m),2.0);
  return C4 + C5/pow((sqrt(m2+C3)+m),2.0) ;
}


/* normalized function, 0 <= x <= 1 */
double Unsteady_Burn::Fxn(double x){
  double Ts = Tmin + (Tmax - Tmin)*x;
  return Ts - Fxn_Ts(Ts);
}


/* the Max value of f_Ts(Ts) */
double Unsteady_Burn::Ts_max(){
  return 0.5*(2.0*R*C2 - Ec + sqrt(4.0*R*R*C2*C2+Ec*Ec))/R;
} 


/* absolute stopping criterion */
int Unsteady_Burn::Termination(){  
  if(fabs(R0 - L0) <= EPS)
    return 1;
  else
    return 0;
}


/* secant method */
double Unsteady_Burn::Secant(double u, double w){  
  double fu = Fxn(u);
  double fw = Fxn(w);
  if(fu != fw)
    return u - fu * (u - w)/(fu - fw); 
  else 
    return UNDEFINED;  
  /* indicates an undefined number in case of fu == fw */
}


void Unsteady_Burn::SetInterval(double x){  
  /* Li = negative,  Ri = positive */
  L3 = L2;
  R3 = R2;
  
  L2 = L1;
  R2 = R1;
  
  L1 = L0;
  R1 = R0;
  
  double y = Fxn(x);
  if(y > 0)       R0 = x;
  else if(y < 0)  L0 = x;
  if(y == 0)      L0 = R0 = x; 
  
  return;
}


double Unsteady_Burn::Bisection(double l, double r){   
  return (l + r)/2;
}


/* Bisection - Secant Method */
double Unsteady_Burn::BisectionSecant(){  
  double a0 = Fxn(0);	  /* a0 < 0 */
  double b0 = Fxn(1);	  /* b0 > 0 */
  double x0 = a0 > -b0 ? 0 : 1;
  double x1 = 0;
  
  while(Termination() == 0){
    
    /* two steps of Regula Falsi */
    for(int j = 1; j <= 2; j++){      
      x1 = x0;
      x0 = Secant(L0, R0);      
      SetInterval(x0);      
      if (Termination() == 1){
	return (L0 + R0)/2;
      }
    }
    
    /* Secant Step */
    while((R0 - L0) <= (R3 - L3)/2){
      double z = Secant(x0, x1); 			
      if((z < L0)||(z > R0)||(UNDEFINED == z))
	/* if z not in (Li, Ri) or z not defined */
	break;      
      x1 = x0;
      x0 = z;      
      SetInterval(x0);      
      if (Termination() == 1)
	return (L0 + R0)/2;
    }
    
    /* Bisection Step */
    x1 = x0;
    x0 = Bisection(L0, R0);
    SetInterval(x0);
  }
  
  return x0;
}
