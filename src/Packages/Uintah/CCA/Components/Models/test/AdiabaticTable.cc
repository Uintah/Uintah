
#include <Packages/Uintah/CCA/Components/ICE/ICEMaterial.h>
#include <Packages/Uintah/CCA/Components/ICE/BoundaryCond.h>
#include <Packages/Uintah/CCA/Components/ICE/Diffusion.h>
#include <Packages/Uintah/CCA/Components/Models/test/AdiabaticTable.h>
#include <Packages/Uintah/CCA/Components/Models/test/TableFactory.h>
#include <Packages/Uintah/CCA/Components/Models/test/TableInterface.h>
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
//  setenv SCI_DEBUG "SIMPLE_RXN_DOING_COUT:+,SIMPLE_RXN_DBG_COUT:+"
//  SIMPLE_RXN_DBG:  dumps out during problemSetup 
//  SIMPLE_RXN_DOING_COUT:   dumps when tasks are scheduled and performed
static DebugStream cout_doing("SIMPLE_RXN_DOING_COUT", false);
static DebugStream cout_dbg("SIMPLE_RXN_DBG_COUT", false);
/*`==========TESTING==========*/
static DebugStream oldStyleAdvect("oldStyleAdvect",false); 
/*==========TESTING==========`*/
//______________________________________________________________________              
AdiabaticTable::AdiabaticTable(const ProcessorGroup* myworld, 
                     ProblemSpecP& params)
  : ModelInterface(myworld), params(params)
{
  d_matl_set = 0;
  lb  = scinew ICELabel();
}


//__________________________________
AdiabaticTable::~AdiabaticTable()
{
  if(d_matl_set && d_matl_set->removeReference()) {
    delete d_matl_set;
  }
  
  VarLabel::destroy(d_scalar->scalar_CCLabel);
  VarLabel::destroy(d_scalar->scalar_source_CCLabel);
  VarLabel::destroy(d_scalar->diffusionCoefLabel);
  delete lb;
  delete table;
  for(int i=0;i<(int)tablevalues.size();i++)
    delete tablevalues[i];
  
  for(vector<Region*>::iterator iter = d_scalar->regions.begin();
                                iter != d_scalar->regions.end(); iter++){
    Region* region = *iter;
    delete region->piece;
    delete region;

  }
}

//__________________________________
AdiabaticTable::Region::Region(GeometryPiece* piece, ProblemSpecP& ps)
  : piece(piece)
{
  ps->require("scalar", initialScalar);
}
//______________________________________________________________________
//     P R O B L E M   S E T U P
void AdiabaticTable::problemSetup(GridP&, SimulationStateP& in_state,
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


  cout_doing << "Doing problemSetup \t\t\t\tSIMPLE_RXN" << endl;
  sharedState = in_state;
  d_matl = sharedState->parseAndLookupMaterial(params, "material");

  vector<int> m(1);
  m[0] = d_matl->getDWIndex();
  d_matl_set = new MaterialSet();
  d_matl_set->addAll(m);
  d_matl_set->addReference();

  string tablename = "adiabatic";
  table = TableFactory::readTable(params, tablename);
  table->addIndependentVariable("mix. frac.");
  for (ProblemSpecP child = params->findBlock("tableValue"); child != 0;
       child = child->findNextBlock("tableValue")) {
    TableValue* tv = new TableValue;
    child->get(tv->name);
    tv->index = table->addDependentVariable(tv->name);
    string labelname = tablename+"-"+tv->name;
    tv->label = VarLabel::create(labelname, CCVariable<double>::getTypeDescription());
    tablevalues.push_back(tv);
  }
  temp_index = table->addDependentVariable("Temp(K)");
  density_index = table->addDependentVariable("density (kg/m3)");
  gamma_index = table->addDependentVariable("gamma");
  cv_index = table->addDependentVariable("heat capacity Cv(j/kg-K)");
  viscosity_index = table->addDependentVariable("viscosity");
  table->setup();
  
  //__________________________________
  // - create Label names
  // - Let ICE know that this model computes the 
  //   thermoTransportProperties.
  // - register the scalar to be transported
  d_scalar = new Scalar();
  d_scalar->index = 0;
  d_scalar->name  = "f";
  
  const TypeDescription* td_CCdouble = CCVariable<double>::getTypeDescription();
  d_scalar->scalar_CCLabel =     VarLabel::create("scalar-f",       td_CCdouble);
  d_scalar->diffusionCoefLabel = VarLabel::create("scalar-diffCoef",td_CCdouble);
  d_scalar->scalar_source_CCLabel = 
                                 VarLabel::create("scalar-f_src",  td_CCdouble);  
  d_modelComputesThermoTransportProps = true;
  
  setup->registerTransportedVariable(d_matl_set->getSubset(0),
                                     d_scalar->scalar_CCLabel,
                                     d_scalar->scalar_source_CCLabel);  
  //__________________________________
  // Read in the constants for the scalar
   ProblemSpecP child = params->findBlock("scalar");
   if (!child){
     throw ProblemSetupException("AdiabaticTable: Couldn't find scalar tag");    
   }
   ProblemSpecP const_ps = child->findBlock("constants");
   if(!const_ps) {
     throw ProblemSetupException("AdiabaticTable: Couldn't find constants tag");
   }
    
   const_ps->get("diffusivity",      d_scalar->diff_coeff);

  //__________________________________
  //  Read in the geometry objects for the scalar
  for (ProblemSpecP geom_obj_ps = child->findBlock("geom_object");
    geom_obj_ps != 0;
    geom_obj_ps = geom_obj_ps->findNextBlock("geom_object") ) {
    vector<GeometryPiece*> pieces;
    GeometryPieceFactory::create(geom_obj_ps, pieces);

    GeometryPiece* mainpiece;
    if(pieces.size() == 0){
     throw ParameterNotFound("No piece specified in geom_object");
    } else if(pieces.size() > 1){
     mainpiece = scinew UnionGeometryPiece(pieces);
    } else {
     mainpiece = pieces[0];
    }

    d_scalar->regions.push_back(scinew Region(mainpiece, geom_obj_ps));
  }
  if(d_scalar->regions.size() == 0) {
    throw ProblemSetupException("Variable: scalar-f does not have any initial value regions");
  }

  //__________________________________
  //  Read in probe locations for the scalar field
  ProblemSpecP probe_ps = child->findBlock("probePoints");
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
void AdiabaticTable::scheduleInitialize(SchedulerP& sched,
                                   const LevelP& level,
                                   const ModelInfo*)
{
  cout_doing << "SIMPLERXN::scheduleInitialize " << endl;
  Task* t = scinew Task("AdiabaticTable::initialize", this, &AdiabaticTable::initialize);

  t->modifies(lb->sp_vol_CCLabel);
  t->modifies(lb->rho_micro_CCLabel);
  t->modifies(lb->rho_CCLabel);
  t->modifies(lb->specific_heatLabel);
  t->modifies(lb->gammaLabel);
  t->modifies(lb->thermalCondLabel);
  t->modifies(lb->viscosityLabel);
  
  t->computes(d_scalar->scalar_CCLabel);
  
  sched->addTask(t, level->eachPatch(), d_matl_set);
}
//______________________________________________________________________
//       I N I T I A L I Z E
void AdiabaticTable::initialize(const ProcessorGroup*, 
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
    CCVariable<double> rho_micro;
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
    setBC(f,"scalar-f", patch, sharedState,indx, new_dw); 

    vector<constCCVariable<double> > ind_vars;
    ind_vars.push_back(f);
    table->interpolate(density_index, rho_CC, patch->getExtraCellIterator(),
                       ind_vars);
    table->interpolate(gamma_index, gamma, patch->getExtraCellIterator(),
                       ind_vars);
    table->interpolate(cv_index, cv, patch->getExtraCellIterator(),
                       ind_vars);
    table->interpolate(viscosity_index, viscosity, patch->getExtraCellIterator(),
                       ind_vars);
    for(CellIterator iter = patch->getExtraCellIterator();!iter.done();iter++){
      const IntVector& c = *iter;
      rho_micro[c] = rho_CC[c];
      sp_vol[c] = 1./rho_CC[c];
      thermalCond[c] = 0;
    }

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
                    d_probePts[i].x(),d_probePts[i].y(), d_probePts[i].x(),
                    cell.x(), cell.y(), cell.z() );
            fclose(fp);
        }
      }  // loop over probes
    }  // if using probe points
  }  // patches
}

//______________________________________________________________________     
void AdiabaticTable::scheduleModifyThermoTransportProperties(SchedulerP& sched,
                                                   const LevelP& level,
                                                   const MaterialSet* /*ice_matls*/)
{

  cout_doing << "SIMPLE_RXN::scheduleModifyThermoTransportProperties" << endl;

  Task* t = scinew Task("AdiabaticTable::modifyThermoTransportProperties", 
                   this,&AdiabaticTable::modifyThermoTransportProperties);
                   
  t->requires(Task::OldDW, d_scalar->scalar_CCLabel, Ghost::None,0);  
  t->modifies(lb->specific_heatLabel);
  t->modifies(lb->gammaLabel);
  t->modifies(lb->thermalCondLabel);
  t->modifies(lb->viscosityLabel);
  //t->computes(d_scalar->diffusionCoefLabel);
  sched->addTask(t, level->eachPatch(), d_matl_set);
}
//______________________________________________________________________
// Purpose:  Compute the thermo and transport properties.  This gets
//           called at the top of the timestep.
// TO DO:   FIGURE OUT A WAY TO ONLY COMPUTE CV ONCE
void AdiabaticTable::modifyThermoTransportProperties(const ProcessorGroup*, 
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
    
    //new_dw->allocateAndPut(diffusionCoeff, 
    //d_scalar->diffusionCoefLabel,indx, patch);  
    
    new_dw->getModifiable(gamma,       lb->gammaLabel,        indx,patch);
    new_dw->getModifiable(cv,          lb->specific_heatLabel,indx,patch);
    new_dw->getModifiable(thermalCond, lb->thermalCondLabel,  indx,patch);
    new_dw->getModifiable(viscosity,   lb->viscosityLabel,    indx,patch);
    
    old_dw->get(f_old,  d_scalar->scalar_CCLabel,  indx, patch, Ghost::None,0);
    
    vector<constCCVariable<double> > ind_vars;
    ind_vars.push_back(f_old);
    table->interpolate(gamma_index, gamma, patch->getExtraCellIterator(),
                       ind_vars);
    table->interpolate(cv_index, cv, patch->getExtraCellIterator(),
                       ind_vars);
    table->interpolate(viscosity_index, viscosity, patch->getExtraCellIterator(),
                       ind_vars);
    for(CellIterator iter = patch->getExtraCellIterator();!iter.done();iter++){
      const IntVector& c = *iter;
      thermalCond[c] = 0;
    }
    //diffusionCoeff.initialize(d_scalar->diff_coeff);
  }
} 

//______________________________________________________________________
// Purpose:  Compute the specific heat at time.  This gets called immediately
//           after (f) is advected
//  TO DO:  FIGURE OUT A WAY TO ONLY COMPUTE CV ONCE
void AdiabaticTable::computeSpecificHeat(CCVariable<double>& cv_new,
                                    const Patch* patch,
                                    DataWarehouse* new_dw,
                                    const int indx)
{ 
  cout_doing << "Doing computeSpecificHeat on patch "<<patch->getID()<< "\t SIMPLERXN" << endl;

  int test_indx = d_matl->getDWIndex();
  //__________________________________
  //  Compute cv for only one matl.
  if (test_indx != indx)
    return;

  constCCVariable<double> f;
  new_dw->get(f,  d_scalar->scalar_CCLabel,  indx, patch, Ghost::None,0);
  vector<constCCVariable<double> > ind_vars;
  ind_vars.push_back(f);
  table->interpolate(cv_index, cv_new, patch->getExtraCellIterator(),
                     ind_vars);
} 

//______________________________________________________________________
void AdiabaticTable::scheduleMassExchange(SchedulerP& sched,
                              const LevelP& level,
                              const ModelInfo* mi)
{
  cout_doing << "SIMPLE_RXN::scheduleMassExchange" << endl;
  Task* t = scinew Task("AdiabaticTable::massExchange", 
                   this,&AdiabaticTable::massExchange, mi);

  t->requires(Task::OldDW, mi->density_CCLabel,  Ghost::None);
  t->modifies(mi->mass_source_CCLabel);
  t->modifies(mi->sp_vol_source_CCLabel);
  sched->addTask(t, level->eachPatch(), d_matl_set);
}
//______________________________________________________________________
void AdiabaticTable::massExchange(const ProcessorGroup*, 
                             const PatchSubset* patches,
                             const MaterialSubset*,
                             DataWarehouse* old_dw,
                             DataWarehouse* new_dw,
                             const ModelInfo* mi)
{
  delt_vartype delT;
  const Level* level = getLevel(patches);
  old_dw->get(delT, mi->delT_Label,level);
  
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    int indx = d_matl->getDWIndex();
    cout_doing << "Doing massExchange on patch "<<patch->getID()<< "\t\t\t\t SIMPLERXN" << endl;
    CCVariable<double> mass_src, sp_vol_src;
    new_dw->getModifiable(mass_src,   mi->mass_source_CCLabel,  indx,patch);
    new_dw->getModifiable(sp_vol_src, mi->sp_vol_source_CCLabel,indx,patch);
    // current does nothing.
  }
}
//______________________________________________________________________
void AdiabaticTable::scheduleMomentumAndEnergyExchange(SchedulerP& sched,
                                          const LevelP& level,
                                          const ModelInfo* mi)
{
  cout_doing << "SIMPLE_RXN::scheduleMomentumAndEnergyExchange " << endl;
  Task* t = scinew Task("AdiabaticTable::momentumAndEnergyExchange", 
                   this,&AdiabaticTable::momentumAndEnergyExchange, mi);
                     
  Ghost::GhostType  gn = Ghost::None;  
  Ghost::GhostType  gac = Ghost::AroundCells;
  //t->requires(Task::NewDW, d_scalar->diffusionCoefLabel, gac,1);
  t->requires(Task::OldDW, d_scalar->scalar_CCLabel,     gac,1); 
  t->requires(Task::OldDW, mi->density_CCLabel,          gn);
  t->requires(Task::OldDW, mi->temperature_CCLabel,      gn);

  Task::DomainSpec oims = Task::OutOfDomain;  //outside of ice matlSet.
  
  MaterialSubset* press_matl = scinew MaterialSubset();
  press_matl->add(0);
  press_matl->addReference();
  t->requires(Task::NewDW, lb->press_equil_CCLabel, press_matl, oims, gn );
  //t->requires(Task::NewDW, lb->specific_heatLabel,       gn);
  //t->requires(Task::NewDW, lb->gammaLabel, gn);
  //t->requires(Task::OldDW, mi->delT_Label); turn off for AMR
  
  t->modifies(mi->energy_source_CCLabel);
  t->modifies(d_scalar->scalar_source_CCLabel);

  // Interpolated table values
  for(int i=0;i<(int)tablevalues.size();i++){
    TableValue* tv = tablevalues[i];
    t->computes(tv->label);
  }
  sched->addTask(t, level->eachPatch(), d_matl_set);
}

//______________________________________________________________________
void AdiabaticTable::momentumAndEnergyExchange(const ProcessorGroup*, 
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
  Ghost::GhostType gac = Ghost::AroundCells; 
  
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    cout_doing << "Doing momentumAndEnergyExch... on patch "<<patch->getID()<< "\t\tSIMPLERXN" << endl;

    for(int m=0;m<matls->size();m++){
      int matl = matls->get(m);

      // Get mixture fraction, and initialize source to zero
      constCCVariable<double> f_old;
      old_dw->get(f_old,      d_scalar->scalar_CCLabel,    matl, patch, gn, 0);
      CCVariable<double> f_src;
      new_dw->allocateAndPut(f_src, d_scalar->scalar_source_CCLabel,
                             matl, patch);
      f_src.initialize(0);

      // Interpolate out gamma, cv, and temperature
      vector<constCCVariable<double> > ind_vars;
      ind_vars.push_back(f_old);
      CCVariable<double> gamma;
      new_dw->allocateTemporary(gamma, patch);
      table->interpolate(gamma_index, gamma, patch->getExtraCellIterator(),
                         ind_vars);
      CCVariable<double> cv;
      new_dw->allocateTemporary(cv, patch);
      table->interpolate(cv_index, cv, patch->getExtraCellIterator(),
                         ind_vars);
      CCVariable<double> flameTemp;
      new_dw->allocateTemporary(flameTemp, patch);
      table->interpolate(temp_index, flameTemp, patch->getExtraCellIterator(),
                         ind_vars);

      // Get density, temperature, and energy source
      constCCVariable<double> rho_CC;
      old_dw->get(rho_CC, mi->density_CCLabel,          matl, patch, gn, 0);
      constCCVariable<double> oldTemp;
      old_dw->get(oldTemp,     mi->temperature_CCLabel, matl, patch, gn, 0);
      constCCVariable<double> press;
      new_dw->get(press,     lb->press_equil_CCLabel, 0, patch, gn, 0);
      CCVariable<double> energySource;
      new_dw->getModifiable(energySource,   
                            mi->energy_source_CCLabel,  matl, patch);

      Vector dx = patch->dCell();
      double volume = dx.x()*dx.y()*dx.z();
      double maxTemp = 0;
      double maxIncrease = 0;
      double maxDecrease = 0;
      double totalEnergy = 0;
      for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
        IntVector c = *iter;
        double mass = rho_CC[c]*volume;
        double cp = gamma[c] * cv[c];
        double newTemp = flameTemp[c]*press[c]/101325;
        //double newTemp = flameTemp[c];
        double energyx =( newTemp - oldTemp[c]) * cp * mass;
        energySource[c] += energyx;
        //        cerr << c << ", f=" << f_old[c] << ", flameTemp=" << flameTemp[c] << ", press=" << press[c] << ", newTemp=" << newTemp << ", oldTemp=" << oldTemp[c] << ", dtemp=" << newTemp-oldTemp[c] << '\n';
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
#if 0
      //__________________________________
      //  Tack on diffusion
      double diff_coeff_test = d_scalar->diff_coeff;
      if(diff_coeff_test != 0.0){ 
        
        bool use_vol_frac = false; // don't include vol_frac in diffusion calc.
        constCCVariable<double> placeHolder;
        
        scalarDiffusionOperator(new_dw, patch, use_vol_frac,
                                placeHolder, placeHolder,  f_old,
                                f_src, diff_coeff, delT);
      }
#endif

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
      
      // Compute miscellaneous table quantities
      for(int i=0;i<(int)tablevalues.size();i++){
        TableValue* tv = tablevalues[i];
        cerr << "interpolating " << tv->name << '\n';
        CCVariable<double> value;
        new_dw->allocateAndPut(value, tv->label, matl, patch);
        CellIterator iter = patch->getCellIterator();
        table->interpolate(tv->index, value, iter, ind_vars);
      }
    }
  }
}

//__________________________________      
void AdiabaticTable::scheduleComputeStableTimestep(SchedulerP&,
                                      const LevelP&,
                                      const ModelInfo*)
{
  // None necessary...
}
