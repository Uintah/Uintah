
#include <Packages/Uintah/CCA/Components/ICE/ICEMaterial.h>
#include <Packages/Uintah/CCA/Components/ICE/BoundaryCond.h>
#include <Packages/Uintah/CCA/Components/Models/test/VorticityConfinement.h>
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

using namespace Uintah;
using namespace std;

//__________________________________
//  To turn on the output
//  setenv SCI_DEBUG "MODELS_DOING_COUT:+,ADIABATIC_TABLE_DBG_COUT:+"
//  ADIABATIC_TABLE_DBG:  dumps out during problemSetup
static DebugStream cout_doing("MODELS_DOING_COUT", false);
/*`==========TESTING==========*/
static DebugStream oldStyleAdvect("oldStyleAdvect",false); 
/*==========TESTING==========`*/
//______________________________________________________________________              
VorticityConfinement::VorticityConfinement(const ProcessorGroup* myworld, 
                     ProblemSpecP& params)
  : ModelInterface(myworld), params(params)
{
  d_matl_set = 0;
}


//__________________________________
VorticityConfinement::~VorticityConfinement()
{
  if(d_matl_set && d_matl_set->removeReference()) {
    delete d_matl_set;
  }
  
}

//______________________________________________________________________
//     P R O B L E M   S E T U P
void VorticityConfinement::problemSetup(GridP&, SimulationStateP& in_state,
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


  cout_doing << "Doing problemSetup \t\t\t\tADIABATIC_TABLE" << endl;
  sharedState = in_state;
  d_matl = sharedState->parseAndLookupMaterial(params, "material");

  vector<int> m(1);
  m[0] = d_matl->getDWIndex();
  d_matl_set = new MaterialSet();
  d_matl_set->addAll(m);
  d_matl_set->addReference();

  //__________________________________
  // Read in the constants for the scalar
  params->require("scale", scale);

  //__________________________________
  //  Read in probe locations for the scalar field
  ProblemSpecP probe_ps = params->findBlock("probePoints");
  if (probe_ps) {
    d_usingProbePts = true;
    probe_ps->require("probeSamplingFreq", d_probeFreq);
     
    Vector location = Vector(0,0,0);
    map<string,string> attr;                    
    for (ProblemSpecP prob_spec = probe_ps->findBlock("location"); prob_spec != 0; 
                      prob_spec = prob_spec->findNextBlock("location")) {
                      
      prob_spec->get(location);
      prob_spec->getAttributes(attr);
      string name = attr["name"];
      
      d_probePts.push_back(location);
      d_probePtsNames.push_back(name);
    }
  } else {
    d_usingProbePts = false;
  }
}
//______________________________________________________________________
//      S C H E D U L E   I N I T I A L I Z E
void VorticityConfinement::scheduleInitialize(SchedulerP& sched,
                                   const LevelP& level,
                                   const ModelInfo*)
{
}

//______________________________________________________________________     
void VorticityConfinement::scheduleModifyThermoTransportProperties(SchedulerP& sched,
                                                   const LevelP& level,
                                                   const MaterialSet* /*ice_matls*/)
{
}

//______________________________________________________________________
// Purpose:  Compute the specific heat at time.  This gets called immediately
//           after (f) is advected
//  TO DO:  FIGURE OUT A WAY TO ONLY COMPUTE CV ONCE
void VorticityConfinement::computeSpecificHeat(CCVariable<double>& cv_new,
                                    const Patch* patch,
                                    DataWarehouse* new_dw,
                                    const int indx)
{ 
} 

//______________________________________________________________________
void VorticityConfinement::scheduleComputeModelSources(SchedulerP& sched,
                                                 const LevelP& level,
                                                 const ModelInfo* mi)
{
  cout_doing << "VorticityConfinemtn::scheduleComputeModelSources " << endl;
  Task* t = scinew Task("VorticityConfinement::computeModelSources", 
                   this,&VorticityConfinement::computeModelSources, mi);
                    
  Ghost::GhostType  gn = Ghost::None;  
  t->requires(Task::OldDW, mi->velocity_CCLabel, Ghost::AroundCells, 2);;
  t->requires(Task::OldDW, mi->density_CCLabel,          gn);

  t->modifies(mi->momentum_source_CCLabel);

  sched->addTask(t, level->eachPatch(), d_matl_set);
}

//______________________________________________________________________
void VorticityConfinement::computeModelSources(const ProcessorGroup*, 
                                         const PatchSubset* patches,
                                         const MaterialSubset* matls,
                                         DataWarehouse* old_dw,
                                         DataWarehouse* new_dw,
                                         const ModelInfo* mi)
{
  delt_vartype delT;
  const Level* level = getLevel(patches);
  old_dw->get(delT, mi->delT_Label, level);
  double delt = delT;
  Ghost::GhostType gn = Ghost::None;
    
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    cout_doing << "Doing momentumAndEnergyExch... on patch "<<patch->getID()
               << "\t\tADIABATIC_TABLE" << endl;

    for(int m=0;m<matls->size();m++){
      int matl = matls->get(m);
      Vector dx(patch->dCell());
      Vector inv_dx(1./dx.x(), 1./dx.y(), 1./dx.z());
      double volume = dx.x()*dx.y()*dx.z();

      constCCVariable<Vector> vel;
      old_dw->get(vel, mi->velocity_CCLabel, matl, patch, Ghost::AroundCells, 2);
      constCCVariable<double> rho_CC;
      old_dw->get(rho_CC,   mi->density_CCLabel,        matl, patch, gn, 0);
      CCVariable<Vector> momsrc;
      new_dw->getModifiable(momsrc, mi->momentum_source_CCLabel, matl, patch);
      CCVariable<double> curlmag;
      new_dw->allocateTemporary(curlmag, patch, Ghost::AroundCells, 1);
      curlmag.initialize(0);
      for(CellIterator iter = patch->getCellIterator();!iter.done(); iter++){
        const IntVector& c = *iter;
        double dvxdy = (vel[c+IntVector(0,1,0)].x()-vel[c-IntVector(0,1,0)].x())*inv_dx.y()*0.5;
        double dvxdz = (vel[c+IntVector(0,0,1)].x()-vel[c-IntVector(0,0,1)].x())*inv_dx.z()*0.5;
        double dvydx = (vel[c+IntVector(1,0,0)].y()-vel[c-IntVector(1,0,0)].y())*inv_dx.x()*0.5;
        double dvydz = (vel[c+IntVector(0,0,1)].y()-vel[c-IntVector(0,0,1)].y())*inv_dx.z()*0.5;
        double dvzdx = (vel[c+IntVector(1,0,0)].z()-vel[c-IntVector(1,0,0)].z())*inv_dx.x()*0.5;
        double dvzdy = (vel[c+IntVector(0,1,0)].z()-vel[c-IntVector(0,1,0)].z())*inv_dx.y()*0.5;
        Vector curl(dvzdy-dvydz, dvxdz-dvzdx, dvydx-dvxdy);
        curlmag[c] = curl.length();

      }

      double totalsrc = 0;
      CellIterator it = patch->getCellIterator();
      IntVector l(it.begin());
      IntVector h(it.end());
      // Don't apply this around the edges because we do not have
      // complete vorticity information
      if(patch->getBCType(Patch::xminus) == Patch::None)
        l += IntVector(1,0,0);
      if(patch->getBCType(Patch::xplus) == Patch::None)
        h -= IntVector(1,0,0);
      if(patch->getBCType(Patch::yminus) == Patch::None)
        l += IntVector(0,1,0);
      if(patch->getBCType(Patch::yplus) == Patch::None)
        h -= IntVector(0,1,0);
      if(patch->getBCType(Patch::zminus) == Patch::None)
        l += IntVector(0,0,1);
      if(patch->getBCType(Patch::zplus) == Patch::None)
        h -= IntVector(0,0,1);
      
      for(CellIterator iter(l,h);!iter.done(); iter++){
        const IntVector& c = *iter;
        double dw_dx = (curlmag[c+IntVector(1,0,0)]-curlmag[c-IntVector(1,0,0)])*0.5*inv_dx.x()*0.5;
        double dw_dy = (curlmag[c+IntVector(0,1,0)]-curlmag[c-IntVector(0,1,0)])*0.5*inv_dx.y()*0.5;
        double dw_dz = (curlmag[c+IntVector(0,0,1)]-curlmag[c-IntVector(0,0,1)])*0.5*inv_dx.z()*0.5;
        Vector dw(dw_dx, dw_dy, dw_dz);
        double length = dw.length()+1e-8;
        dw *= 1./length;

        double dvxdy = (vel[c+IntVector(0,1,0)].x()-vel[c-IntVector(0,1,0)].x())*inv_dx.y()*0.5;
        double dvxdz = (vel[c+IntVector(0,0,1)].x()-vel[c-IntVector(0,0,1)].x())*inv_dx.z()*0.5;
        double dvydx = (vel[c+IntVector(1,0,0)].y()-vel[c-IntVector(1,0,0)].y())*inv_dx.x()*0.5;
        double dvydz = (vel[c+IntVector(0,0,1)].y()-vel[c-IntVector(0,0,1)].y())*inv_dx.z()*0.5;
        double dvzdx = (vel[c+IntVector(1,0,0)].z()-vel[c-IntVector(1,0,0)].z())*inv_dx.x()*0.5;
        double dvzdy = (vel[c+IntVector(0,1,0)].z()-vel[c-IntVector(0,1,0)].z())*inv_dx.y()*0.5;
        Vector curl(dvzdy-dvydz, dvxdz-dvzdx, dvydx-dvxdy);
        Vector f(Cross(dw, curl));
        double mass      = rho_CC[c]*volume;
        momsrc[c] += f*mass*delt*scale;
        totalsrc += momsrc[c].length();
      }

#if 0
      //__________________________________
      //  dump out the probe points
      if (d_usingProbePts ) {
        double time = d_dataArchiver->getCurrentTime();
        double nextDumpTime = oldProbeDumpTime + 1.0/d_probeFreq;
        
        if (time >= nextDumpTime){        // is it time to dump the points
          FILE *fp;
          string udaDir = d_dataArchiver->getOutputLocation();
          IntVector cell_indx;
          
          // loop through all the points and dump if that patch contains them
          for (unsigned int i =0 ; i < d_probePts.size(); i++) {
            if(patch->findCell(Point(d_probePts[i]),cell_indx) ) {
              string filename=udaDir + "/" + d_probePtsNames[i].c_str() + ".dat";
              fp = fopen(filename.c_str(), "a");
              fprintf(fp, "%16.15E  %16.15E\n",time, f_old[cell_indx]);
              fclose(fp);
            }
          }
          oldProbeDumpTime = time;
        }  // time to dump
      } // if(probePts)  
#endif
      
    }
  }
}

//__________________________________      
void VorticityConfinement::scheduleComputeStableTimestep(SchedulerP&,
                                      const LevelP&,
                                      const ModelInfo*)
{
  // None necessary...
}
//______________________________________________________________________
//
void VorticityConfinement::scheduleErrorEstimate(const LevelP&,
                                           SchedulerP&)
{
  // Not implemented yet
}
