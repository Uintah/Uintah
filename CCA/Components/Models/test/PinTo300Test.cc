
#include <Packages/Uintah/CCA/Components/Models/test/PinTo300Test.h>
#include <Packages/Uintah/CCA/Components/ICE/ICEMaterial.h>
#include <Packages/Uintah/CCA/Components/ICE/BoundaryCond.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Material.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Packages/Uintah/Core/Grid/GeomPiece/GeometryPieceFactory.h>
#include <Packages/Uintah/Core/Grid/GeomPiece/UnionGeometryPiece.h>
#include <Packages/Uintah/Core/Labels/ICELabel.h>
#include <Packages/Uintah/Core/Exceptions/ParameterNotFound.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/Core/Exceptions/InvalidValue.h>
#include <Core/Math/MiscMath.h>
#include <iostream>
#include <Core/Util/DebugStream.h>
#include <stdio.h>

// TODO:
// 1. Call modifyThermo from intialize instead of duping code
// Drive density...

using namespace Uintah;
using namespace std;
//__________________________________
//  To turn on the output
//  setenv SCI_DEBUG "MODELS_DOING_COUT:+,PINTO300TEST_DBG_COUT:+"
//  PINTO300TEST_DBG:  dumps out during problemSetup 
static DebugStream cout_doing("MODELS_DOING_COUT", false);
static DebugStream cout_dbg("PINTO300TEST_DBG_COUT", false);
/*`==========TESTING==========*/
static DebugStream oldStyleAdvect("oldStyleAdvect",false); 
/*==========TESTING==========`*/
//______________________________________________________________________              
PinTo300Test::PinTo300Test(const ProcessorGroup* myworld, 
                     ProblemSpecP& params)
  : ModelInterface(myworld), params(params)
{
  d_matl_set = 0;
  lb  = scinew ICELabel();
}


//__________________________________
PinTo300Test::~PinTo300Test()
{
  if(d_matl_set && d_matl_set->removeReference()) {
    delete d_matl_set;
  }
  
  delete lb;
}

//______________________________________________________________________
//     P R O B L E M   S E T U P
void PinTo300Test::problemSetup(GridP&, SimulationStateP& in_state,
                        ModelSetup* setup)
{
/*`==========TESTING==========*/
if (!oldStyleAdvect.active()){
  ostringstream desc;
  desc<< "\n----------------------------\n"
      <<" ICE need the following environmental variable \n"
       << " \t setenv SCI_DEBUG oldStyleAdvect:+ \n"
       << "for this model to work.  This is gross--Todd"
       << "\n----------------------------\n";
  throw ProblemSetupException(desc.str());  
} 
/*==========TESTING==========`*/


  cout_doing << "Doing problemSetup \t\t\t\tPINTO300TEST" << endl;
  sharedState = in_state;
  d_matl = sharedState->parseAndLookupMaterial(params, "material");

  vector<int> m(1);
  m[0] = d_matl->getDWIndex();
  d_matl_set = new MaterialSet();
  d_matl_set->addAll(m);
  d_matl_set->addReference();

  d_modelComputesThermoTransportProps = false;
}

//______________________________________________________________________
//      S C H E D U L E   I N I T I A L I Z E
void PinTo300Test::scheduleInitialize(SchedulerP& sched,
                                   const LevelP& level,
                                   const ModelInfo*)
{
  cout_doing << "PINTO300TEST::scheduleInitialize " << endl;
  Task* t = scinew Task("PinTo300Test::initialize", this, &PinTo300Test::initialize);

  t->modifies(lb->sp_vol_CCLabel);
  t->modifies(lb->rho_micro_CCLabel);
  t->modifies(lb->rho_CCLabel);
  t->modifies(lb->specific_heatLabel);
  t->modifies(lb->thermalCondLabel);
  t->modifies(lb->viscosityLabel);
  
  sched->addTask(t, level->eachPatch(), d_matl_set);
}
//______________________________________________________________________
//       I N I T I A L I Z E
void PinTo300Test::initialize(const ProcessorGroup*, 
                           const PatchSubset* patches,
                           const MaterialSubset*,
                           DataWarehouse*,
                           DataWarehouse* new_dw)
{
  cout_doing << "Doing Initialize \t\t\t\t\tPINTO300TEST" << endl;
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    int indx = d_matl->getDWIndex();
    
    CCVariable<double>  cv, thermalCond, viscosity, rho_CC, sp_vol;
    CCVariable<double> rho_micro;
    constCCVariable<double> Temp;
    new_dw->getModifiable(rho_CC,      lb->rho_CCLabel,       indx,patch);
    new_dw->getModifiable(sp_vol,      lb->sp_vol_CCLabel,    indx,patch);
    new_dw->getModifiable(rho_micro,   lb->rho_micro_CCLabel, indx,patch);
    new_dw->getModifiable(cv,          lb->specific_heatLabel,indx,patch);
    new_dw->getModifiable(thermalCond, lb->thermalCondLabel,  indx,patch);
    new_dw->getModifiable(viscosity,   lb->viscosityLabel,    indx,patch);

    //__________________________________
    //  Dump out a header for the probe point files
    oldProbeDumpTime = 0;
    if (d_usingProbePts){
      FILE *fp;
      IntVector cell;
      string udaDir = d_dataArchiver->getOutputLocation();
      
        for (unsigned int i =0 ; i < d_probePts.size(); i++) {
          if(patch->findCell(Point(d_probePts[i]),cell) ) {
            string filename=udaDir + "/" + d_probePtsNames[i].c_str() + ".dat";
            fp = fopen(filename.c_str(), "a");
            fprintf(fp, "%%Time Scalar Field at [%e, %e, %e], at cell [%i, %i, %i]\n", 
                    d_probePts[i].x(),d_probePts[i].y(), d_probePts[i].z(),
                    cell.x(), cell.y(), cell.z() );
            fclose(fp);
        }
      }  // loop over probes
    }  // if using probe points
  }  // patches
}

//______________________________________________________________________     
void PinTo300Test::scheduleModifyThermoTransportProperties(SchedulerP& sched,
                                                   const LevelP& level,
                                                   const MaterialSet* /*ice_matls*/)
{
  // Nothing
}

//______________________________________________________________________
// Purpose:  Compute the specific heat at time.  This gets called immediately
//           after (f) is advected
//  TO DO:  FIGURE OUT A WAY TO ONLY COMPUTE CV ONCE
void PinTo300Test::computeSpecificHeat(CCVariable<double>& cv_new,
                                    const Patch* patch,
                                    DataWarehouse* new_dw,
                                    const int indx)
{ 
  cout_doing << "Doing computeSpecificHeat on patch "<<patch->getID()<< "\t PINTO300TEST" << endl;

  int test_indx = d_matl->getDWIndex();
  //__________________________________
  //  Compute cv for only one matl.
  if (test_indx != indx)
    return;

  cerr << "!!!computeSpecificHeat busted\n";
#if 0
  constCCVariable<double> f;
  new_dw->get(f,  d_scalar->scalar_CCLabel,  indx, patch, Ghost::None,0);
  vector<constCCVariable<double> > ind_vars;
  ind_vars.push_back(f);
  table->interpolate(cv_index, cv_new, patch->getExtraCellIterator(),
                     ind_vars);
#endif
} 


//______________________________________________________________________
void PinTo300Test::scheduleComputeModelSources(SchedulerP& sched,
                                               const LevelP& level,
                                               const ModelInfo* mi)
{
  cout_doing << "PINTO300TEST::scheduleComputeModelSources " << endl;
  Task* t = scinew Task("PinTo300Test::computeModelSources", 
                   this,&PinTo300Test::computeModelSources, mi);
                     
  Ghost::GhostType  gn = Ghost::None;  
  Task::DomainSpec oims = Task::OutOfDomain;  //outside of ice matlSet.
  
  MaterialSubset* press_matl = scinew MaterialSubset();
  press_matl->add(0);
  press_matl->addReference();
  
  //t->requires(Task::NewDW, d_scalar->diffusionCoefLabel, gac,1);
  t->requires(Task::OldDW, mi->density_CCLabel,          gn);
  t->requires(Task::OldDW, mi->temperature_CCLabel,      gn);
  t->requires(Task::NewDW, lb->press_equil_CCLabel, press_matl, oims, gn );
  //t->requires(Task::NewDW, lb->specific_heatLabel,       gn);
  //t->requires(Task::OldDW, mi->delT_Label); turn off for AMR
  
  t->modifies(mi->energy_source_CCLabel);

  sched->addTask(t, level->eachPatch(), d_matl_set);
}

//______________________________________________________________________
void PinTo300Test::computeModelSources(const ProcessorGroup*, 
                                       const PatchSubset* patches,
                                       const MaterialSubset* matls,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw,
                                       const ModelInfo* mi)
{
  delt_vartype delT;
  const Level* level = getLevel(patches);
  old_dw->get(delT, mi->delT_Label, level);
  Ghost::GhostType gn = Ghost::None;         
    
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    cout_doing << "Doing computeModelSources... on patch "<<patch->getID()
               << "\t\tPinTo300Test" << endl;

    for(int m=0;m<matls->size();m++){
      int matl = matls->get(m);

      // Get density, temperature, and energy source
      constCCVariable<double> rho_CC;
      constCCVariable<double> oldTemp;
      constCCVariable<double> press;
      CCVariable<double> energySource;
      old_dw->get(rho_CC,  mi->density_CCLabel,     matl, patch, gn, 0);
      old_dw->get(oldTemp, mi->temperature_CCLabel, matl, patch, gn, 0);
      new_dw->get(press,   lb->press_equil_CCLabel,0,     patch, gn, 0);
      new_dw->getModifiable(energySource,   
                            mi->energy_source_CCLabel,  matl, patch);

      Vector dx = patch->dCell();
      double volume = dx.x()*dx.y()*dx.z();
      double maxTemp = 0;
      double maxIncrease = 0;
      double maxDecrease = 0;
      double totalEnergy = 0;
      double gamma = 1.4;
      double cv = 716.5;
      double cp = gamma*cv;
      double flameTemp = 300.0;
      
      
      for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
        IntVector c = *iter;
        double mass = rho_CC[c]*volume;
        double newTemp = flameTemp*press[c]/101325;
        
        double energyx =( newTemp - oldTemp[c]) * cp * mass;
        energySource[c] += energyx;
        totalEnergy += energyx;
        
        
        if(newTemp > maxTemp)
          maxTemp = newTemp;
        double dtemp = newTemp-oldTemp[c];
        if(dtemp > maxIncrease)
          maxIncrease = dtemp;
        if(dtemp < maxDecrease)
          maxDecrease = dtemp;
      }
      cerr << "MaxTemp = " << maxTemp << ", maxIncrease=" << maxIncrease << ", maxDecrease=" << maxDecrease << ", totalEnergy=" << totalEnergy << '\n';
    }
  }
}

//__________________________________      
void PinTo300Test::scheduleComputeStableTimestep(SchedulerP&,
                                      const LevelP&,
                                      const ModelInfo*)
{
  // None necessary...
}
//______________________________________________________________________
//
void PinTo300Test::scheduleErrorEstimate(const LevelP&,
                                         SchedulerP&)
{
  // Not implemented yet
}
