/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
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


#include <CCA/Components/ICE/ICEMaterial.h>
#include <CCA/Components/ICE/ConservationTest.h>
#include <CCA/Components/ICE/BoundaryCond.h>
#include <CCA/Components/ICE/Diffusion.h>
#include <CCA/Components/Models/FluidsBased/SimpleRxn.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/Box.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Material.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/GeometryPiece/GeometryPieceFactory.h>
#include <Core/GeometryPiece/UnionGeometryPiece.h>
#include <Core/Exceptions/ParameterNotFound.h>
#include <Core/Labels/ICELabel.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Containers/StaticArray.h>
#include <Core/Math/MiscMath.h>
#include <iostream>
#include <Core/Util/DebugStream.h>
#include <cstdio>

using namespace Uintah;
using namespace std;
//__________________________________
//  To turn on the output
//  setenv SCI_DEBUG "MODELS_DOING_COUT:+,SIMPLE_RXN_DBG_COUT:+"
//  SIMPLE_RXN_DBG:  dumps out during problemSetup 
static DebugStream cout_doing("MODELS_DOING_COUT", false);
static DebugStream cout_dbg("SIMPLE_RXN_DBG_COUT", false);
//______________________________________________________________________              
SimpleRxn::SimpleRxn(const ProcessorGroup* myworld, 
                     ProblemSpecP& params)
  : ModelInterface(myworld), params(params)
{
  d_matl_set = 0;
  lb  = scinew ICELabel();
  Slb = scinew SimpleRxnLabel();
}

//__________________________________
SimpleRxn::~SimpleRxn()
{
  if(d_matl_set && d_matl_set->removeReference()) {
    delete d_matl_set;
  }
  
  VarLabel::destroy(d_scalar->scalar_CCLabel);
  VarLabel::destroy(d_scalar->scalar_source_CCLabel);
  VarLabel::destroy(d_scalar->diffusionCoefLabel);
  VarLabel::destroy(Slb->lastProbeDumpTimeLabel);
  VarLabel::destroy(Slb->sum_scalar_fLabel);
  delete lb;
  delete Slb;
  
  for(vector<Region*>::iterator iter = d_scalar->regions.begin();
                                iter != d_scalar->regions.end(); iter++){
    Region* region = *iter;
    delete region;

  }
}

void SimpleRxn::outputProblemSpec(ProblemSpecP& ps)
{

}

//__________________________________
SimpleRxn::Region::Region(GeometryPieceP piece, ProblemSpecP& ps)
  : piece(piece)
{
  ps->require("scalar", initialScalar);
}
//______________________________________________________________________
//     P R O B L E M   S E T U P
void SimpleRxn::problemSetup(GridP&, SimulationStateP& in_state,
                        ModelSetup* setup)
{
  cout_doing << "Doing problemSetup \t\t\t\tSIMPLE_RXN" << endl;
  sharedState = in_state;
  d_matl = sharedState->parseAndLookupMaterial(params, "material");

  vector<int> m(1);
  m[0] = d_matl->getDWIndex();
  d_matl_set = scinew MaterialSet();
  d_matl_set->addAll(m);
  d_matl_set->addReference();
  
  //__________________________________
  // - create Label names
  // - Let ICE know that this model computes the 
  //   thermoTransportProperties.
  // - register the scalar to be transported
  d_scalar = scinew Scalar();
  d_scalar->index = 0;
  d_scalar->name  = "f";
  
  const TypeDescription* td_CCdouble = CCVariable<double>::getTypeDescription();
  d_scalar->scalar_CCLabel =     VarLabel::create("scalar-f",       td_CCdouble);
  d_scalar->diffusionCoefLabel = VarLabel::create("scalar-diffCoef",td_CCdouble);
  d_scalar->scalar_source_CCLabel = 
                                 VarLabel::create("scalar-f_src",  td_CCdouble);
  Slb->lastProbeDumpTimeLabel =  VarLabel::create("lastProbeDumpTime", 
                                            max_vartype::getTypeDescription());
  Slb->sum_scalar_fLabel      =  VarLabel::create("sum_scalar_f", 
                                            sum_vartype::getTypeDescription());                         
  
  d_modelComputesThermoTransportProps = true;
  
  setup->registerTransportedVariable(d_matl_set,
                                     d_scalar->scalar_CCLabel,
                                     d_scalar->scalar_source_CCLabel);  
  //__________________________________
  // Read in the constants for the scalar
   ProblemSpecP child = params->findBlock("SimpleRxn")->findBlock("scalar");
   if (!child){
     throw ProblemSetupException("SimpleRxn: Couldn't find (SimpleRxn) or (scalar) tag", __FILE__, __LINE__);    
   }
   child->getWithDefault("test_conservation", d_test_conservation, false);
   
   ProblemSpecP const_ps = child->findBlock("constants");
   if(!const_ps) {
     throw ProblemSetupException("SimpleRxn: Couldn't find constants tag", __FILE__, __LINE__);
   }
    
  const_ps->getWithDefault("f_stoichometric",d_scalar->f_stoic,    -9);         

  const_ps->getWithDefault("rho_air",          d_rho_air,         -9);
  const_ps->getWithDefault("rho_fuel",         d_rho_fuel,        -9);
  const_ps->getWithDefault("cv_air",           d_cv_air,          -9);
  const_ps->getWithDefault("cv_fuel",          d_cv_fuel,         -9);
  const_ps->getWithDefault("R_air",            d_R_air,           -9);
  const_ps->getWithDefault("R_fuel",           d_R_fuel,          -9);
  const_ps->getWithDefault("thermalCond_air",  d_thermalCond_air,  0);
  const_ps->getWithDefault("thermalCond_fuel", d_thermalCond_fuel, 0);
  const_ps->getWithDefault("viscosity_air",    d_viscosity_air,    0);
  const_ps->getWithDefault("viscosity_fuel",   d_viscosity_fuel,   0);
  const_ps->getWithDefault("diffusivity",      d_scalar->diff_coeff, -9);       
  const_ps->getWithDefault("initialize_diffusion_knob",       
                            d_scalar->initialize_diffusion_knob,   0);
  
  if( d_scalar->f_stoic == -9 || 
      d_rho_air   == -9  || d_rho_fuel == -9 ||    
      d_cv_air    == -9  || d_cv_fuel  == -9 ||  
      d_R_air     == -9  || d_R_fuel   == -9  ) {
    ostringstream warn;
    warn << " ERROR SimpleRxn: Input variable(s) not specified \n" 
         << "\n f_stoichometric  "<< d_scalar->f_stoic 
         << "\n diffusivity      "<< d_scalar->diff_coeff
         << "\n rho_air          "<< d_rho_air
         << "\n rho_fuel         "<< d_rho_fuel
         << "\n R_fuel           "<< d_R_fuel
         << "\n R_air            "<< d_R_air<<endl;
    throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
  } 

  //__________________________________
  //  Read in the geometry objects for the scalar
  for (ProblemSpecP geom_obj_ps = child->findBlock("geom_object");
    geom_obj_ps != 0;
    geom_obj_ps = geom_obj_ps->findNextBlock("geom_object") ) {
    vector<GeometryPieceP> pieces;
    GeometryPieceFactory::create(geom_obj_ps, pieces);

    GeometryPieceP mainpiece;
    if(pieces.size() == 0){
     throw ParameterNotFound("No piece specified in geom_object", __FILE__, __LINE__);
    } else if(pieces.size() > 1){
     mainpiece = scinew UnionGeometryPiece(pieces);
    } else {
     mainpiece = pieces[0];
    }

    d_scalar->regions.push_back(scinew Region(mainpiece, geom_obj_ps));
  }
  if(d_scalar->regions.size() == 0) {
    throw ProblemSetupException("Variable: scalar-f does not have any initial value regions", __FILE__, __LINE__);
  }

  //__________________________________
  //  Read in probe locations for the scalar field
  ProblemSpecP probe_ps = child->findBlock("probePoints");
  d_usingProbePts = true;
  
  if (probe_ps) {
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
      d_usingProbePts = true;
    }
  } 
}
//______________________________________________________________________
//      S C H E D U L E   I N I T I A L I Z E
void SimpleRxn::scheduleInitialize(SchedulerP& sched,
                                   const LevelP& level,
                                   const ModelInfo*)
{
  cout_doing << "SIMPLERXN::scheduleInitialize " << endl;
  Task* t = scinew Task("SimpleRxn::initialize", this, &SimpleRxn::initialize);

  t->modifies(lb->sp_vol_CCLabel);
  t->modifies(lb->rho_micro_CCLabel);
  t->modifies(lb->rho_CCLabel);
  t->modifies(lb->specific_heatLabel);
  t->modifies(lb->gammaLabel);
  t->modifies(lb->thermalCondLabel);
  t->modifies(lb->viscosityLabel);
  t->modifies(lb->press_CCLabel);
  
  t->computes(d_scalar->scalar_CCLabel);
  t->computes(Slb->lastProbeDumpTimeLabel);
  
  sched->addTask(t, level->eachPatch(), d_matl_set);
}
//______________________________________________________________________
//       I N I T I A L I Z E
void SimpleRxn::initialize(const ProcessorGroup*, 
                           const PatchSubset* patches,
                           const MaterialSubset*,
                           DataWarehouse*,
                           DataWarehouse* new_dw)
{
  cout_doing << "Doing Initialize \t\t\t\t\tSIMPLE_RXN" << endl;
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    int indx = d_matl->getDWIndex();
    
    CCVariable<double>  f, cv, gamma, thermalCond, viscosity, rho_CC, sp_vol;
    CCVariable<double> press, rho_micro;
    constCCVariable<double> Temp;
    new_dw->allocateAndPut(f, d_scalar->scalar_CCLabel, indx, patch);
    new_dw->getModifiable(rho_CC,      lb->rho_CCLabel,       indx,patch);
    new_dw->getModifiable(sp_vol,      lb->sp_vol_CCLabel,    indx,patch);
    new_dw->getModifiable(rho_micro,   lb->rho_micro_CCLabel, indx,patch);
    new_dw->getModifiable(gamma,       lb->gammaLabel,        indx,patch);
    new_dw->getModifiable(cv,          lb->specific_heatLabel,indx,patch);
    new_dw->getModifiable(thermalCond, lb->thermalCondLabel,  indx,patch);
    new_dw->getModifiable(viscosity,   lb->viscosityLabel,    indx,patch);
    //__________________________________
    //  initialize the scalar field in a region
    f.initialize(0);

    for(vector<Region*>::iterator iter = d_scalar->regions.begin();
                                  iter != d_scalar->regions.end(); iter++){
      Region* region = *iter;
      for(CellIterator iter = patch->getExtraCellIterator();
          !iter.done(); iter++){
        IntVector c = *iter;
        Point p = patch->cellPosition(c);            
        if(region->piece->inside(p)) {
          f[c] = region->initialScalar;
        }
      } // Over cells
    } // regions

    //__________________________________
    //  Smooth out initial distribution with some diffusion
    // WARNING::This may have problems on multiple patches
    //          look at the initial distribution where the patches meet
    CCVariable<double> FakeDiffusivity;
    new_dw->allocateTemporary(FakeDiffusivity, patch, Ghost::AroundCells, 1);
    FakeDiffusivity.initialize(1.0);    //  HARDWIRED
    double fakedelT = 1.0;
    
    for( int i =1 ; i < d_scalar->initialize_diffusion_knob; i++ ){
      bool use_vol_frac = false; // don't include vol_frac in diffusion calc.
      CCVariable<double> placeHolder;
      scalarDiffusionOperator(new_dw, patch, use_vol_frac, f, placeHolder,
                              f, FakeDiffusivity, fakedelT);
    }
        
    setBC(f,"scalar-f", patch, sharedState,indx, new_dw); 
    
    //__________________________________
    // compute thermo-transport-physical quantities
    // This MUST be identical to what's in the task modifyThermoTransport....
    for(CellIterator iter = patch->getExtraCellIterator();!iter.done(); iter++){
      IntVector c = *iter;
      double oneMinus_f = 1.0 - f[c];       
      cv[c]          = f[c]*d_cv_fuel          + oneMinus_f*d_cv_air;           
      viscosity[c]   = f[c]*d_viscosity_fuel   + oneMinus_f*d_viscosity_air;    
      thermalCond[c] = f[c]*d_thermalCond_fuel + oneMinus_f*d_thermalCond_air; 
      double R_mix   = f[c]*d_R_fuel           + oneMinus_f*d_R_air;           
      gamma[c]       = R_mix/cv[c]  + 1.0;

      rho_CC[c]      = d_rho_fuel * d_rho_air /
                      ( f[c] * d_rho_air + oneMinus_f * d_rho_fuel);
      rho_micro[c]   = rho_CC[c];
      sp_vol[c]      = 1.0/rho_CC[c];
    } // Over cells

    //__________________________________
    //  Dump out a header for the probe point files
    new_dw->put(max_vartype(0.0), Slb->lastProbeDumpTimeLabel);
    if (d_usingProbePts){
      FILE *fp;
      IntVector cell;
      string udaDir = d_dataArchiver->getOutputLocation();
      
        for (unsigned int i =0 ; i < d_probePts.size(); i++) {
          if(patch->findCell(Point(d_probePts[i]),cell) ) {
            string filename=udaDir + "/" + d_probePtsNames[i].c_str() + ".dat";
            fp = fopen(filename.c_str(), "a");
            fprintf(fp, "%% Time Scalar Field at [%e, %e, %e], at cell [%i, %i, %i]\n", 
                    d_probePts[i].x(),d_probePts[i].y(), d_probePts[i].z(),
                    cell.x(), cell.y(), cell.z() );
            fclose(fp);
        }
      }  // loop over probes
    }  // if using probe points
  }  // patches
}

//______________________________________________________________________     
void SimpleRxn::scheduleModifyThermoTransportProperties(SchedulerP& sched,
                                                   const LevelP& level,
                                                   const MaterialSet* /*ice_matls*/)
{

  cout_doing << "SIMPLE_RXN::scheduleModifyThermoTransportProperties" << endl;

  Task* t = scinew Task("SimpleRxn::modifyThermoTransportProperties", 
                   this,&SimpleRxn::modifyThermoTransportProperties);
                   
  t->requires(Task::OldDW, d_scalar->scalar_CCLabel, Ghost::None,0);  
  t->modifies(lb->specific_heatLabel);
  t->modifies(lb->gammaLabel);
  t->modifies(lb->thermalCondLabel);
  t->modifies(lb->viscosityLabel);
  t->computes(d_scalar->diffusionCoefLabel);
  sched->addTask(t, level->eachPatch(), d_matl_set);
}
//______________________________________________________________________
// Purpose:  Compute the thermo and transport properties.  This gets
//           called at the top of the timestep.
// TO DO:   FIGURE OUT A WAY TO ONLY COMPUTE CV ONCE
void SimpleRxn::modifyThermoTransportProperties(const ProcessorGroup*, 
                                                const PatchSubset* patches,
                                                const MaterialSubset*,
                                                DataWarehouse* old_dw,
                                                DataWarehouse* new_dw)
{ 
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    cout_doing << "Doing modifyThermoTransportProperties on patch "<<patch->getID()<< "\t SIMPLERXN" << endl;
   
    int indx = d_matl->getDWIndex();
    CCVariable<double> diffusionCoeff, gamma, cv, thermalCond, viscosity;
    constCCVariable<double> f_old;
    
    new_dw->allocateAndPut(diffusionCoeff, 
                           d_scalar->diffusionCoefLabel,indx, patch);  
    
    new_dw->getModifiable(gamma,       lb->gammaLabel,        indx,patch);
    new_dw->getModifiable(cv,          lb->specific_heatLabel,indx,patch);
    new_dw->getModifiable(thermalCond, lb->thermalCondLabel,  indx,patch);
    new_dw->getModifiable(viscosity,   lb->viscosityLabel,    indx,patch);
    
    old_dw->get(f_old,  d_scalar->scalar_CCLabel,  indx, patch, Ghost::None,0);
    
    //__________________________________
    for(CellIterator iter = patch->getExtraCellIterator(); !iter.done(); iter++){
      IntVector c = *iter;
      double  f = f_old[c];
      double oneMinus_f = 1.0 - f;       
      cv[c]          = f * d_cv_fuel          + oneMinus_f*d_cv_air;          
      viscosity[c]   = f * d_viscosity_fuel   + oneMinus_f*d_viscosity_air;   
      thermalCond[c] = f * d_thermalCond_fuel + oneMinus_f*d_thermalCond_air; 
      double R_mix   = f * d_R_fuel           + oneMinus_f*d_R_air;
      gamma[c]       = R_mix/cv[c]  + 1.0;
    }     
    diffusionCoeff.initialize(d_scalar->diff_coeff);
  }
} 

//______________________________________________________________________
// Purpose:  Compute the specific heat at time.  This gets called immediately
//           after (f) is advected
//  TO DO:  FIGURE OUT A WAY TO ONLY COMPUTE CV ONCE
void SimpleRxn::computeSpecificHeat(CCVariable<double>& cv_new,
                                    const Patch* patch,
                                    DataWarehouse* new_dw,
                                    const int indx)
{ 
  cout_doing << "Doing computeSpecificHeat on patch "<<patch->getID()<< "\t SIMPLERXN" << endl;

  int test_indx = d_matl->getDWIndex();
  //__________________________________
  //  Compute cv for only one matl.
  if (test_indx == indx) {
    constCCVariable<double> f;
    new_dw->get(f,  d_scalar->scalar_CCLabel,  indx, patch, Ghost::None,0);

    for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
      IntVector c = *iter;
      cv_new[c]  = f[c] * d_cv_fuel + (1.0 - f[c])*d_cv_air;
    }
  }
} 

//______________________________________________________________________
void SimpleRxn::scheduleComputeModelSources(SchedulerP& sched,
                                            const LevelP& level,
                                            const ModelInfo* mi)
{
  cout_doing << "SIMPLE_RXN::scheduleComputeModelSources " << endl;
  Task* t = scinew Task("SimpleRxn::computeModelSources", 
                   this,&SimpleRxn::computeModelSources, mi);
                     
  Ghost::GhostType  gn = Ghost::None;  
  Ghost::GhostType  gac = Ghost::AroundCells;
  t->requires(Task::OldDW, mi->delT_Label, level.get_rep());
  t->requires(Task::NewDW, d_scalar->diffusionCoefLabel, gac,1);
  t->requires(Task::OldDW, d_scalar->scalar_CCLabel,     gac,1); 
  t->requires(Task::OldDW, mi->rho_CCLabel,    gn);
  t->requires(Task::OldDW, mi->temp_CCLabel,   gn);
  
  
  t->modifies(mi->modelMom_srcLabel);
  t->modifies(mi->modelEng_srcLabel);
  t->modifies(d_scalar->scalar_source_CCLabel);
  //__________________________________
  //  if dumping out probePts
  if (d_usingProbePts){
    t->requires(Task::OldDW, Slb->lastProbeDumpTimeLabel);
    t->computes(Slb->lastProbeDumpTimeLabel);
  }
  sched->addTask(t, level->eachPatch(), d_matl_set);
}

//______________________________________________________________________
void SimpleRxn::computeModelSources(const ProcessorGroup*, 
                                    const PatchSubset* patches,
                                    const MaterialSubset* /*matls*/,
                                    DataWarehouse* old_dw,
                                    DataWarehouse* new_dw,
                                    const ModelInfo* mi)
{
  const Level* level = getLevel(patches);
  delt_vartype delT;
  old_dw->get(delT, mi->delT_Label, level);
  Ghost::GhostType gn = Ghost::None;         
  Ghost::GhostType gac = Ghost::AroundCells; 
  
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    cout_doing << "Doing computeModelSources... on patch "<<patch->getID()
               << "\t\tSIMPLERXN" << endl;

    double new_f;
    constCCVariable<double> rho_CC_old,f_old, diff_coeff;
    CCVariable<double> eng_src, sp_vol_src, f_src;
    CCVariable<Vector> mom_src;
    
    int indx = d_matl->getDWIndex();
    old_dw->get(rho_CC_old, mi->rho_CCLabel,               indx, patch, gn, 0);
    old_dw->get(f_old,      d_scalar->scalar_CCLabel,      indx, patch, gac,1);
    new_dw->get(diff_coeff, d_scalar->diffusionCoefLabel,  indx, patch, gac,1);
    new_dw->allocateAndPut(f_src, d_scalar->scalar_source_CCLabel,
                                                           indx, patch);
                                                      
    new_dw->getModifiable(mom_src,  mi->modelMom_srcLabel,indx,patch);
    new_dw->getModifiable(eng_src,  mi->modelEng_srcLabel,indx,patch);

    //__________________________________
    // rho=1/(f/rho_fuel+(1-f)/rho_air)
    double fuzzyOne = 1.0 + 1e-10;
    double fuzzyZero = 0.0 - 1e10;
    int     numCells = 0, sum = 0;
    double f_stoic= d_scalar->f_stoic;

    //__________________________________   
    for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
      IntVector c = *iter;
      //double mass_old = rho_CC_old[c]*volume;
      double f = f_old[c];
      numCells++;

      f = min(f,1.0);    // keep 0 < f < 1
      f = max(f,0.0);
      //__________________________________
      //  Inside the flame    
      if (f_stoic < f && f <= fuzzyOne ){  
        sum++;
      }                     // Does nothing right now
      //__________________________________
      //  At the flame surface        
      if (f_stoic == f ){ 
        sum++;
      }
      //__________________________________
      //  outside the flame
      if (fuzzyZero <= f && f < f_stoic ){ 
         sum++;
      }  
/*`==========TESTING==========*/
  new_f = f;   // hardwired off
/*==========TESTING==========`*/
      f_src[c] += new_f - f;
    }  //iter

    //__________________________________
    //  bulletproofing
    if (sum != numCells) {
      ostringstream warn;
      warn << "ERROR: SimpleRxn Model: invalid value for f "
           << "somewhere in the scalar field: "<< sum
           << " cells were touched out of "<< numCells
           << " Total cells ";
      throw InvalidValue(warn.str(), __FILE__, __LINE__);
    }     
    //__________________________________
    //  Tack on diffusion
    double diff_coeff_test = d_scalar->diff_coeff;
    if(diff_coeff_test != 0.0){ 
      
      bool use_vol_frac = false; // don't include vol_frac in diffusion calc.
      CCVariable<double> placeHolder;

      scalarDiffusionOperator(new_dw, patch, use_vol_frac, f_old,
                              placeHolder,f_src, diff_coeff, delT);
    }


    //__________________________________
    //  dump out the probe points
    if (d_usingProbePts ) {
      
      max_vartype lastDumpTime;
      old_dw->get(lastDumpTime, Slb->lastProbeDumpTimeLabel);
      double oldProbeDumpTime = lastDumpTime;
      
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
      new_dw->put(max_vartype(oldProbeDumpTime), Slb->lastProbeDumpTimeLabel);
    } // if(probePts)  
  }
}
//______________________________________________________________________
void SimpleRxn::scheduleTestConservation(SchedulerP& sched,
                                         const PatchSet* patches,
                                         const ModelInfo* mi)
{
  if(d_test_conservation){
    cout_doing << "SIMPLE_RXN::scheduleTestConservation " << endl;
    Task* t = scinew Task("SimpleRxn::testConservation", 
                     this,&SimpleRxn::testConservation, mi);

    Ghost::GhostType  gn = Ghost::None;
    t->requires(Task::OldDW, mi->delT_Label, getLevel(patches) );
    t->requires(Task::NewDW, d_scalar->scalar_CCLabel, gn,0); 
    t->requires(Task::NewDW, mi->rho_CCLabel,          gn,0);
    t->requires(Task::NewDW, lb->uvel_FCMELabel,       gn,0); 
    t->requires(Task::NewDW, lb->vvel_FCMELabel,       gn,0); 
    t->requires(Task::NewDW, lb->wvel_FCMELabel,       gn,0); 
    t->computes(Slb->sum_scalar_fLabel);

    sched->addTask(t, patches, d_matl_set);
  }
}

//______________________________________________________________________
void SimpleRxn::testConservation(const ProcessorGroup*, 
                                 const PatchSubset* patches,
                                 const MaterialSubset* /*matls*/,
                                 DataWarehouse* old_dw,
                                 DataWarehouse* new_dw,
                                 const ModelInfo* mi)
{
  const Level* level = getLevel(patches);
  delt_vartype delT;
  old_dw->get(delT, mi->delT_Label, level);     
  Ghost::GhostType gn = Ghost::None; 
  
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    cout_doing << "Doing testConservation on patch "<<patch->getID()
               << "\t\t\t SimpleRxn" << endl;
               
    //__________________________________
    //  conservation of f test
    constCCVariable<double> rho_CC, f;
    constSFCXVariable<double> uvel_FC;
    constSFCYVariable<double> vvel_FC;
    constSFCZVariable<double> wvel_FC;
    int indx = d_matl->getDWIndex();
    new_dw->get(f,       d_scalar->scalar_CCLabel,indx,patch,gn,0);
    new_dw->get(rho_CC,  mi->rho_CCLabel,         indx,patch,gn,0); 
    new_dw->get(uvel_FC, lb->uvel_FCMELabel,      indx,patch,gn,0); 
    new_dw->get(vvel_FC, lb->vvel_FCMELabel,      indx,patch,gn,0); 
    new_dw->get(wvel_FC, lb->wvel_FCMELabel,      indx,patch,gn,0); 
    Vector dx = patch->dCell();
    double cellVol = dx.x()*dx.y()*dx.z();

    CCVariable<double> q_CC;
    new_dw->allocateTemporary(q_CC, patch);

    for(CellIterator iter = patch->getExtraCellIterator(); !iter.done(); iter++) {
      IntVector c = *iter;
      q_CC[c] = rho_CC[c]*cellVol*f[c];
    }

    double sum_mass_f;
    conservationTest<double>(patch, delT, q_CC, uvel_FC, vvel_FC, wvel_FC, 
                             sum_mass_f);
    
    new_dw->put(sum_vartype(sum_mass_f), Slb->sum_scalar_fLabel);
  }
}


//__________________________________      
void SimpleRxn::scheduleComputeStableTimestep(SchedulerP&,
                                      const LevelP&,
                                      const ModelInfo*)
{
  // None necessary...
}
//______________________________________________________________________
//
void SimpleRxn::scheduleErrorEstimate(const LevelP&,
                                       SchedulerP&)
{
  // Not implemented yet
}
