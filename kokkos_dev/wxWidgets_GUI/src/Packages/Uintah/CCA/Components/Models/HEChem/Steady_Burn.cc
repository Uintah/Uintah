#include <Packages/Uintah/CCA/Components/Models/HEChem/Steady_Burn.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/NCVariable.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/Variables/VarTypes.h>
#include <Packages/Uintah/Core/Labels/MPMLabel.h>
#include <Packages/Uintah/Core/Labels/ICELabel.h>
#include <Packages/Uintah/Core/Labels/MPMICELabel.h>
#include <Packages/Uintah/CCA/Components/ICE/ICEMaterial.h>
#include <Packages/Uintah/CCA/Components/ICE/BoundaryCond.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Core/Util/DebugStream.h>

#include <iostream>

#undef DEBUG_SCALAR
#define DEBUG_SCALAR

#undef TOTALS
//#define TOTALS    


using namespace Uintah;
using namespace SCIRun;
using namespace std;
//__________________________________   
//  setenv SCI_DEBUG "MPMICE_NORMAL_COUT:+,MPMICE_DOING_COUT:+"
//  MPMICE_DOING_COUT:   dumps when tasks are scheduled and performed
static DebugStream cout_doing("MPMICE_DOING_COUT", false);


const double Steady_Burn::R = 8.314;
const double Steady_Burn::EPS = 1.e-5; 
const double Steady_Burn::UNDEFINED = -10; 

Steady_Burn::Steady_Burn(const ProcessorGroup* myworld, ProblemSpecP& params)
  : ModelInterface(myworld), params(params) { 
  mymatls = 0;
  Mlb  = scinew MPMLabel();
  Ilb  = scinew ICELabel();
  MIlb = scinew MPMICELabel();

#ifdef DEBUG_SCALAR 
  SomeScalarLabel = VarLabel::create("SteadyBurn.SomeScalar", CCVariable<double>::getTypeDescription());
#endif
}


Steady_Burn::~Steady_Burn(){
  delete Ilb;
  delete Mlb;
  delete MIlb;

#ifdef DEBUG_SCALAR
  VarLabel::destroy(SomeScalarLabel);
#endif

  if(mymatls && mymatls->removeReference())
    delete mymatls;
}


void Steady_Burn::problemSetup(GridP&, SimulationStateP& sharedState, ModelSetup*){
  cout<<"I am in problem setup" << endl;
  d_sharedState = sharedState;
  matl0 = sharedState->parseAndLookupMaterial(params, "fromMaterial");
  matl1 = sharedState->parseAndLookupMaterial(params, "toMaterial");  
  params->require("PreExpCondPh",      Ac);
  params->require("ActEnergyCondPh",   Ec);
  params->require("PreExpGasPh",       Bg);
  params->require("CondPhaseHeat",     Qc);
  params->require("GasPhaseHeat",      Qg);
  params->require("HeatConductGasPh",  Kg);
  params->require("HeatConductCondPh", Kc);
  params->require("SpecificHeatBoth",  Cp);
  params->require("MoleWeightGasPh",   MW);
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
  mymatls = new MaterialSet();            
  
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

void Steady_Burn::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP model_ps = ps->appendChild("Model",true,3);
  model_ps->setAttribute("type","Steady_Burn");

  model_ps->appendElement("fromMaterial",matl0->getName(),false,4);
  model_ps->appendElement("toMaterial",matl1->getName(),false,4);
  model_ps->appendElement("PreExpCondPh",      Ac,false,4);
  model_ps->appendElement("ActEnergyCondPh",   Ec,false,4);
  model_ps->appendElement("PreExpGasPh",       Bg,false,4);
  model_ps->appendElement("CondPhaseHeat",     Qc,false,4);
  model_ps->appendElement("GasPhaseHeat",      Qg,false,4);
  model_ps->appendElement("HeatConductGasPh",  Kg,false,4);
  model_ps->appendElement("HeatConductCondPh", Kc,false,4);
  model_ps->appendElement("SpecificHeatBoth",  Cp,false,4);
  model_ps->appendElement("MoleWeightGasPh",   MW,false,4);
  model_ps->appendElement("IgnitionTemp",      ignitionTemp,false,4);
}

void Steady_Burn::scheduleInitialize(SchedulerP& sched, const LevelP& level, const ModelInfo*){
  cout_doing << "Steady_Burn::scheduleInitialize" << endl;
  Task* t = scinew Task("Steady_Burn::initialize", this, &Steady_Burn::initialize);                        
  MaterialSubset* one_matl  = scinew MaterialSubset();
  one_matl->add(0);
  one_matl->addReference();     
  sched->addTask(t, level->eachPatch(), mymatls);
}


void Steady_Burn::initialize(const ProcessorGroup*, 
			     const PatchSubset* patches, 
			     const MaterialSubset* /*matls*/, 
			     DataWarehouse*, 
			     DataWarehouse* new_dw){
  for(int p=0;p<patches->size();p++) {
    const Patch* patch = patches->get(p);
    cout_doing << "Doing Initialize on patch " << patch->getID()<< "\t\t\t STEADY_BURN" << endl; 
  }        
}


void Steady_Burn::scheduleComputeStableTimestep(SchedulerP&, const LevelP&, const ModelInfo*){
  // None necessary...
}


void Steady_Burn::scheduleComputeModelSources(SchedulerP& sched, const LevelP& level, const ModelInfo* mi){
  Task* t = scinew Task("Steady_Burn::computeModelSources", this, &Steady_Burn::computeModelSources, mi);
  cout_doing << "Steady_Burn::scheduleComputeModelSources" << endl;
  
  t->requires( Task::OldDW, mi->delT_Label);

  Ghost::GhostType  gac = Ghost::AroundCells;  
  Ghost::GhostType  gn  = Ghost::None;

  const MaterialSubset* react_matl = matl0->thisMaterial();
  //const MaterialSubset* prod_matl  = matl1->thisMaterial();
  
  MaterialSubset* one_matl     = scinew MaterialSubset();
  one_matl->add(0);
  one_matl->addReference();
  MaterialSubset* press_matl   = one_matl;

  /*
    const MaterialSubset* ice_matls = d_sharedState->allICEMaterials()->getUnion();
    const MaterialSubset* mpm_matls = d_sharedState->allMPMMaterials()->getUnion();
  */
  
  t->requires(Task::OldDW, Ilb->temp_CCLabel,                  gac,1);
  /* t->requires(Task::NewDW, Ilb->temp_CCLabel,      mpm_matls,  gac,1); */
  t->requires(Task::NewDW, Ilb->vol_frac_CCLabel,              gac,1);

  /*     Products     */
  
  /*     Reactants    */
  t->requires(Task::NewDW, Ilb->sp_vol_CCLabel,   react_matl, gn);
  t->requires(Task::NewDW, MIlb->vel_CCLabel,     react_matl, gn);
  t->requires(Task::NewDW, MIlb->cMassLabel,      react_matl, gn);
  t->requires(Task::NewDW, MIlb->gMassLabel,      react_matl, gac,1);
  t->requires(Task::OldDW, Mlb->pXLabel,          react_matl, gac,1);
  
  /*     Misc      */
  t->requires(Task::NewDW,  Ilb->press_equil_CCLabel, press_matl, gn);
  t->requires(Task::OldDW,  MIlb->NC_CCweightLabel,   one_matl,   gac, 1);  
   
  t->modifies(mi->mass_source_CCLabel);
  t->modifies(mi->momentum_source_CCLabel);
  t->modifies(mi->energy_source_CCLabel);
  t->modifies(mi->sp_vol_source_CCLabel); 

#ifdef DEBUG_SCALAR
  t->computes(SomeScalarLabel, react_matl);  
#endif
   
  sched->addTask(t, level->eachPatch(), mymatls);
  
  if(one_matl->removeReference())
    delete one_matl;
}


void Steady_Burn::scheduleModifyThermoTransportProperties(SchedulerP&, const LevelP&, const MaterialSet*){
  // do nothing      
}


void Steady_Burn::computeSpecificHeat(CCVariable<double>&, const Patch*, DataWarehouse*, const int){
  //do nothing
}



/*
 ***************** Private Member Functions:******************************
 */
void Steady_Burn::computeModelSources(const ProcessorGroup*, 
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
  
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);  
    
    cout_doing << "Doing massExchange on patch "<< patch->getID()<<"\t\t\t\t Steady_Burn"<<endl;
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
    
    constParticleVariable<Point>  px;
    CCVariable<double> PartFlag;/* for indicating cells containing particles */

#ifdef DEBUG_SCALAR
    CCVariable<double> SomeScalar;
#endif    

    Ghost::GhostType  gn  = Ghost::None;    
    Ghost::GhostType  gac = Ghost::AroundCells;
    
    /* Reactant data */
    old_dw->get(solidTemp,       MIlb->temp_CCLabel,    m0, patch, gac, 1);
    new_dw->get(solidMass,       MIlb->cMassLabel,      m0, patch, gn, 0);
    new_dw->get(solidSp_vol,     Ilb->sp_vol_CCLabel,   m0, patch, gn, 0);   
    new_dw->get(vel_CC,          MIlb->vel_CCLabel,     m0, patch, gn, 0);
    new_dw->get(NCsolidMass,     MIlb->gMassLabel,      m0, patch, gac,1);

    ParticleSubset* pset = old_dw->getParticleSubset(m0, patch, gac,1, Mlb->pXLabel);
    old_dw->get(px, Mlb->pXLabel, pset);
    
    /* Product Data */
       
    /* Misc */
    new_dw->get(press_CC,       Ilb->press_equil_CCLabel,      0, patch, gn,  0);
    old_dw->get(NC_CCweight,    MIlb->NC_CCweightLabel,        0, patch, gac, 1);

#ifdef DEBUG_SCALAR
    new_dw->allocateAndPut(SomeScalar, Steady_Burn::SomeScalarLabel, m0, patch);
#endif

    new_dw->allocateTemporary(PartFlag, patch, gac, 1);

    IntVector nodeIdx[8];

    /* All Material Data */
    int numAllMatls = d_sharedState->getNumMatls();
    StaticArray<constCCVariable<double> >    vol_frac_CC(numAllMatls);
    StaticArray<constCCVariable<double> >    temp_CC(numAllMatls);

    for (int m = 0; m < numAllMatls; m++) {
      Material* matl = d_sharedState->getMaterial(m);
      int indx = matl->getDWIndex();
      old_dw->get(temp_CC[m],       MIlb->temp_CCLabel,      indx, patch, gac, 1);
      new_dw->get(vol_frac_CC[m],   Ilb->vol_frac_CCLabel,   indx, patch, gac, 1);
    }
    
    
    PartFlag.initialize(0.0);/* initialize extra cells for BC to 100 */
    for(ParticleSubset::iterator iter = pset->begin(); iter != pset->end(); iter++){
      particleIndex idx = *iter;
      IntVector c;
      patch->findCell(px[idx],c);
      PartFlag[c] += 1.0;
    }    
    setBC(PartFlag, "set_if_sym_BC", patch, d_sharedState, m0, new_dw);
 
    Vector  dx = patch->dCell();

#ifdef TOTALS
    double totalSurfArea=0.0, totalBurnedMass=0.0;
    int totalNumBurningCells=0;
#endif      

#ifdef DEBUG_SCALAR
    SomeScalar.initialize(0.0);
#endif

    for (CellIterator iter = patch->getCellIterator();!iter.done();iter++){
      IntVector c = *iter;
      patch->findNodesFromCell(*iter,nodeIdx);
      
      bool burning = 0;
      double MaxMass = d_SMALL_NUM;
      double MinMass = 1.0/d_SMALL_NUM; 
      for (int nN=0; nN<8; nN++){
	MaxMass = std::max(MaxMass,NC_CCweight[nodeIdx[nN]]*NCsolidMass[nodeIdx[nN]]);
	MinMass = std::min(MinMass,NC_CCweight[nodeIdx[nN]]*NCsolidMass[nodeIdx[nN]]); 
      }


#if 0
      if(c.x()==18 || c.x()==19 || c.x()==20 || c.x()==21){
	IntVector cell = c + IntVector(0,1,1);
	cout<<"cell = "<<cell<<"   PartFlag ="<<PartFlag[cell]<<"  volFrac="<<vol_frac_CC[0][cell]
	    <<"  temp="<<temp_CC[0][cell]<<endl;
      }
#endif


      /*if( (MaxMass-MinMass)/MaxMass>0.4 && (MaxMass-MinMass)/MaxMass<1.0 && PartFlag[c]>0 ){ */
      if( MinMass/MaxMass<0.7 && PartFlag[c]>0 ){ 
	/* near interface and containing particles */
	for(int i = -1; i<=1; i++){
	  for(int j = -1; j<=1; j++){
	    for(int k = -1; k<=1; k++){
	      IntVector cell = c + IntVector(i,j,k);

#if 0
	      cout<<"cell="<<cell<<"  PF ="<<PartFlag[cell]
		  <<"  vf_0="<<vol_frac_CC[0][cell] <<"  vf_1="<<vol_frac_CC[1][cell]<<"  vf_2="<<vol_frac_CC[2][cell]
		  <<"  t_0="<<temp_CC[0][cell]<<"  t_1="<<temp_CC[1][cell]<<"  t_2="<<temp_CC[2][cell]<<endl;
#endif     
 
	      if( PartFlag[cell] <= 3.0){
		for (int m = 0; m < numAllMatls; m++){
		  if(vol_frac_CC[m][cell] > 0.2 && temp_CC[m][cell] > ignitionTemp){
		    burning = 1;
		    break;
		  }
		}
	      }
	      if(burning) break;
	    }
	    if(burning) break;
	  }
	  if(burning) break;
	}
      }
      
      if(burning){

#ifdef DEBUG_SCALAR
	SomeScalar[c]=1.0;
#endif
       	//cout<<"\tThe burning cell is : " << c << " solid vol_frac = " << vol_frac_CC[0][c] <<"  solidMass = " <<solidMass[c] <<endl;
       	Vector rhoGradVector = computeDensityGradientVector(nodeIdx, NCsolidMass, NC_CCweight,dx);
       	double surfArea = computeSurfaceArea(rhoGradVector, dx); 

	double burnedMass = computeBurnedMass(300.0 /* solidTemp[c] */, press_CC[c], solidSp_vol[c], 
					      surfArea, delT, solidMass[c]);
	
#ifdef TOTALS
	totalSurfArea += surfArea;
	totalBurnedMass += burnedMass;
	totalNumBurningCells++;
#endif

	/* conservation of mass, momentum and energy   */
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

#ifdef TOTALS    
    if(0.0 != totalBurnedMass){
      cout<<"  TotalSurfArea = "<<totalSurfArea <<"   TotalBurnedMass = " <<totalBurnedMass 
	  <<"   TotalNumBurningCells = "<<totalNumBurningCells<<endl;
    }
#endif

    /*  set symetric BC  */
    setBC(mass_src_0, "set_if_sym_BC",patch, d_sharedState, m0, new_dw);
    setBC(mass_src_1, "set_if_sym_BC",patch, d_sharedState, m1, new_dw); 
  }
}

//______________________________________________________________________

double Steady_Burn::computeSurfaceArea(Vector &rhoGradVector, Vector &dx){
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
  
  /* double TmpX = fabs(delX*rhoGradVector.x());
  double TmpY = fabs(delY*rhoGradVector.y());
  double TmpZ = fabs(delZ*rhoGradVector.z());*/
 
  double TmpX = delX*rgvX;
  double TmpY = delY*rgvY;
  double TmpZ = delZ*rgvZ;
    
  return delX*delY*delZ / (TmpX+TmpY+TmpZ) * coeff; 
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

void Steady_Burn::scheduleErrorEstimate(const LevelP&, SchedulerP&){
  // Not implemented yet
}

void Steady_Burn::scheduleTestConservation(SchedulerP&, const PatchSet*, const ModelInfo*){
  // Not implemented yet
}

void Steady_Burn::setMPMLabel(MPMLabel* MLB){
  Mlb = MLB;
}



/****************************************************************************/
/******************* Bisection Secant Solver ********************************/
/****************************************************************************/


double Steady_Burn::computeBurnedMass(double To, double P, double Vc, double surfArea, double delT, double solidMass){  
  UpdateConstants(To, P, Vc);
  double Ts = Tmin + (Tmax - Tmin) * BisectionSecant();
  double Mr =  sqrt(C1*Ts*Ts/(Ts-C2)*exp(-Ec/R/Ts));
  double burnedMass = delT * surfArea * Mr;
  if (burnedMass + d_TINY_RHO > solidMass) 
    burnedMass = solidMass - d_TINY_RHO;  
  return burnedMass;
}


void Steady_Burn::UpdateConstants(double To, double P, double Vc){
  /* CC1 = Ac*R*Kc/Ec/Cp        */
  /* CC2 = Qc/Cp/2              */
  /* CC3 = 4*Kg*Bg*W*W/Cp/R/R;  */
  /* CC4 = Qc/Cp                */
  /* CC5 = Qg/Cp                */
  C1 = CC1 / Vc; /* Vc = Condensed Phase Specific Volume */
  C2 = To + CC2; 
  C3 = CC3 * P*P;
  C4 = To + CC4; 
  C5 = CC5 * C3; 
    
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
double Steady_Burn::Fxn_Ts(double Ts){
  double m2 = C1*Ts*Ts/(Ts-C2)*exp(-Ec/R/Ts);
  double m = sqrt(m2);
  //double result = C4 + C5/pow((sqrt(m2+C3)+m),2.0);
  return C4 + C5/pow((sqrt(m2+C3)+m),2.0) ;
}


/* normalized function, 0 <= x <= 1 */
double Steady_Burn::Fxn(double x){
  double Ts = Tmin + (Tmax - Tmin)*x;
  return Ts - Fxn_Ts(Ts);
}


/* the Max value of f_Ts(Ts) */
double Steady_Burn::Ts_max(){
  return 0.5*(2.0*R*C2 - Ec + sqrt(4.0*R*R*C2*C2+Ec*Ec))/R;
  //cout<< Ts << "  " << f_Ts(Ts) <<endl;
} 


/* absolute stopping criterion */
int Steady_Burn::Termination(){  
  if(fabs(R0 - L0) <= EPS)
    return 1;
  else
    return 0;
}


/* secant method */
double Steady_Burn::Secant(double u, double w){  
  double fu = Fxn(u);
  double fw = Fxn(w);
  if(fu != fw)
    return u - fu * (u - w)/(fu - fw); 
  else 
    return UNDEFINED;  /* indicates an undefined number in case of fu == fw */
}


void Steady_Burn::SetInterval(double x){  
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


double Steady_Burn::Bisection(double l, double r){   
  return (l + r)/2;
}


/* Bisection - Secant Method */
double Steady_Burn::BisectionSecant(){  
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
      if((z < L0)||(z > R0)||(UNDEFINED == z))/* if z not in (Li, Ri) or z not defined */
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
