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

// TODO
// Use cp directly instead of cv/gamma
// multimaterial checks in smagorisky
// multimaterial checks in dynamic model
// Avoid recomputation of variance

#include <CCA/Components/ICE/ICEMaterial.h>
#include <CCA/Components/ICE/ConservationTest.h>
#include <CCA/Components/ICE/BoundaryCond.h>
#include <CCA/Components/ICE/Diffusion.h>
#include <CCA/Components/Models/FluidsBased/AdiabaticTable.h>
#include <CCA/Components/Models/FluidsBased/TableFactory.h>
#include <CCA/Components/Models/FluidsBased/TableInterface.h>
#include <CCA/Components/Regridder/PerPatchVars.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/Box.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Material.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/GeometryPiece/GeometryPieceFactory.h>
#include <Core/GeometryPiece/UnionGeometryPiece.h>
#include <Core/Labels/ICELabel.h>
#include <Core/Exceptions/ParameterNotFound.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Math/MiscMath.h>
#include <iostream>
#include <Core/Util/DebugStream.h>
#include <cstdio>

using namespace Uintah;
using namespace std;

//__________________________________
//  To turn on the output
//  setenv SCI_DEBUG "MODELS_DOING_COUT:+,ADIABATIC_TABLE_DBG_COUT:+"
//  ADIABATIC_TABLE_DBG:  dumps out during problemSetup
static DebugStream cout_doing("MODELS_DOING_COUT", false);
static DebugStream cout_dbg("ADIABATIC_TABLE_DBG_COUT", false);

//______________________________________________________________________              
AdiabaticTable::AdiabaticTable(const ProcessorGroup* myworld, 
                     ProblemSpecP& params,
                     const bool doAMR)
  : ModelInterface(myworld), params(params)
{
  d_doAMR = doAMR;
  d_scalar = 0;
  d_matl_set = 0;
  lb  = scinew ICELabel();
  cumulativeEnergyReleased_CCLabel = VarLabel::create("cumulativeEnergyReleased", CCVariable<double>::getTypeDescription());
  cumulativeEnergyReleased_src_CCLabel = VarLabel::create("cumulativeEnergyReleased_src", CCVariable<double>::getTypeDescription());
}


//__________________________________
AdiabaticTable::~AdiabaticTable()
{
  if(d_matl_set && d_matl_set->removeReference()) {
    delete d_matl_set;
  }
  
  if(d_scalar){
    VarLabel::destroy(d_scalar->scalar_CCLabel);
    VarLabel::destroy(d_scalar->diffusionCoeffLabel);
    VarLabel::destroy(d_scalar->varianceLabel);
    VarLabel::destroy(d_scalar->scaledVarianceLabel);
    VarLabel::destroy(d_scalar->sum_scalar_fLabel);
    VarLabel::destroy(d_scalar->mag_grad_scalarLabel);
      
    if(d_scalar->scalar_src_CCLabel){
      VarLabel::destroy(d_scalar->scalar_src_CCLabel);
    }
  }
  VarLabel::destroy(cumulativeEnergyReleased_CCLabel);
  VarLabel::destroy(cumulativeEnergyReleased_src_CCLabel);
  
  delete lb;
  delete table;
  for(int i=0;i<(int)tablevalues.size();i++){
    TableValue* tv = tablevalues[i];
    VarLabel::destroy(tv->label);
    delete tablevalues[i];
  }
  
  for(vector<Region*>::iterator iter = d_scalar->regions.begin();
                                iter != d_scalar->regions.end(); iter++){
    Region* region = *iter;
    delete region;

  }
  delete d_scalar;
}

//__________________________________
AdiabaticTable::Region::Region(GeometryPieceP piece, ProblemSpecP& ps)
  : piece(piece)
{
  ps->require("scalar", initialScalar);
}

void AdiabaticTable::Region::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP geom_obj_ps = ps->appendChild("geom_object");
  piece->outputProblemSpec(geom_obj_ps);
  geom_obj_ps->appendElement("scalar", initialScalar);
}

void AdiabaticTable::Scalar::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP scalar_ps = ps->appendChild("scalar");
  scalar_ps->setAttribute("name",name);

  scalar_ps->appendElement("test_conservation", d_test_conservation);
  scalar_ps->appendElement("doTableTest", d_doTableTest);

  ProblemSpecP constants_ps = scalar_ps->appendChild("constants");
  constants_ps->appendElement("diffusivity",diff_coeff);
  constants_ps->appendElement("AMR_Refinement_Criteria",
                              refineCriteria);
  
  for (vector<Region*>::const_iterator it = regions.begin();
       it != regions.end(); ++it) {
    (*it)->outputProblemSpec(scalar_ps);
  }

}


void AdiabaticTable::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP model_ps = ps->appendChild("Model");
  model_ps->setAttribute("type","AdiabaticTable");

  for (vector<TableValue*>::const_iterator it = tablevalues.begin();
       it != tablevalues.end(); ++it)
    (*it)->outputProblemSpec(model_ps);

  model_ps->appendElement("material",d_matl->getName());

  model_ps->appendElement("varianceScale",d_varianceScale);
  model_ps->appendElement("varianceMax",  d_varianceMax);

#if 1
  ProblemSpecP table_ps = model_ps->appendChild("table");
  table_ps->setAttribute("name","adiabatic");
  table->outputProblemSpec(table_ps);
#endif
  d_scalar->outputProblemSpec(model_ps);
  
}


//______________________________________________________________________
//     P R O B L E M   S E T U P
void AdiabaticTable::problemSetup(GridP&, SimulationStateP& in_state,
                        ModelSetup* setup)
{
  cout_doing << "Doing problemSetup \t\t\t\tADIABATIC_TABLE" << endl;
  d_sharedState = in_state;
  d_matl = d_sharedState->parseAndLookupMaterial(params, "material");

  vector<int> m(1);
  m[0] = d_matl->getDWIndex();
  d_matl_set = scinew MaterialSet();
  d_matl_set->addAll(m);
  d_matl_set->addReference();

  // Get parameters
  params->getWithDefault("varianceScale", d_varianceScale, 0.0);
  params->getWithDefault("varianceMax",   d_varianceMax, 1.0);
  d_useVariance = (d_varianceScale != 0.0);

  //__________________________________
  //setup the table
  string tablename = "adiabatic";
  table = TableFactory::readTable(params, tablename);
  table->addIndependentVariable("mixture_fraction");
  if(d_useVariance)
    table->addIndependentVariable("mixture_fraction_variance");
  
  for (ProblemSpecP child = params->findBlock("tableValue"); child != 0;
       child = child->findNextBlock("tableValue")) {
    TableValue* tv = scinew TableValue;
    child->get(tv->name);
    tv->index = table->addDependentVariable(tv->name);
    string labelname = tv->name;
    tv->label = VarLabel::create(labelname, CCVariable<double>::getTypeDescription());
    tablevalues.push_back(tv);
  }
  
  d_temp_index          = table->addDependentVariable("temperature");
  d_density_index       = table->addDependentVariable("density");
  d_gamma_index         = table->addDependentVariable("gamma");
  d_cv_index            = table->addDependentVariable("heat_capac_Cv");
  d_viscosity_index     = table->addDependentVariable("viscosity");
  d_thermalcond_index   = table->addDependentVariable("thermal_conductivity");
  d_ref_cv_index    = table->addDependentVariable("reference_heat_capac_Cv");
  d_ref_gamma_index = table->addDependentVariable("reference_gamma");
  d_ref_temp_index  = table->addDependentVariable("reference_Temp");
//  d_MW_index        = table->addDependentVariable("mixture_molecular_weight");
  
  bool cerrSwitch = (d_myworld->myrank() == 0); 
  table->setup(cerrSwitch);

  //__________________________________
  // - create Label names
  // - Let ICE know that this model computes the 
  //   thermoTransportProperties.
  // - register the scalar to be transported
  d_scalar = scinew Scalar();
  d_scalar->index = 0;
  d_scalar->name  = "f";
  
  const TypeDescription* td_CCdouble = CCVariable<double>::getTypeDescription();
  d_scalar->scalar_CCLabel = VarLabel::create("scalar-f",         td_CCdouble);
  d_scalar->varianceLabel  = VarLabel::create("scalar-f-variance",td_CCdouble);
  d_scalar->mag_grad_scalarLabel = 
                                  VarLabel::create("mag_grad_scalar-f",
                                                                  td_CCdouble);
  d_scalar->scaledVarianceLabel = VarLabel::create("scalar-f-scaledvariance", 
                                                                  td_CCdouble);
  d_scalar->diffusionCoeffLabel = VarLabel::create("scalar-diffCoeff",
                                                                  td_CCdouble);
  d_scalar->sum_scalar_fLabel   = VarLabel::create("sum_scalar_f", 
                                            sum_vartype::getTypeDescription()); 
                                            
                                            
  d_modelComputesThermoTransportProps = true;
  
  //__________________________________
  // Read in the constants for the scalar
  ProblemSpecP child = params->findBlock("scalar");
  if (!child){
    throw ProblemSetupException("AdiabaticTable: Couldn't find scalar tag", __FILE__, __LINE__);    
  }
   
  child->getWithDefault("test_conservation", d_scalar->d_test_conservation, false);
  child->getWithDefault("doTableTest",       d_scalar->d_doTableTest, false);
   
  ProblemSpecP const_ps = child->findBlock("constants");
  if(!const_ps) {
    throw ProblemSetupException("AdiabaticTable:Couldn't find constants tag", __FILE__, __LINE__);
  }
 
  const_ps->getWithDefault("diffusivity",  d_scalar->diff_coeff, 0.0);
  const_ps->getWithDefault("AMR_Refinement_Criteria", d_scalar->refineCriteria,1e100);

  // scalar src is only needed if we have non-zero diffusivity
  if(d_scalar->diff_coeff){
    d_scalar->scalar_src_CCLabel = VarLabel::create("scalar-f_src",td_CCdouble);
  }else{
    d_scalar->scalar_src_CCLabel = 0;
  }

  //__________________________________
  //  register transport variables scalar and the energy
  setup->registerTransportedVariable(d_matl_set,
                                     d_scalar->scalar_CCLabel,
                                     d_scalar->scalar_src_CCLabel);
  setup->registerTransportedVariable(d_matl_set,
                                     cumulativeEnergyReleased_CCLabel,
                                     cumulativeEnergyReleased_src_CCLabel);

  //__________________________________
  //  register the AMRrefluxing variables                               
  if(d_doAMR){
   setup->registerAMR_RefluxVariable(d_matl_set,
                                     d_scalar->scalar_CCLabel);

   setup->registerAMR_RefluxVariable(d_matl_set,
                                     cumulativeEnergyReleased_CCLabel);
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
void AdiabaticTable::scheduleInitialize(SchedulerP& sched,
                                   const LevelP& level,
                                   const ModelInfo*)
{
  cout_doing << "ADIABATIC_TABLE::scheduleInitialize " << endl;
  Task* t = scinew Task("AdiabaticTable::initialize", this, 
                        &AdiabaticTable::initialize);

  t->modifies(lb->sp_vol_CCLabel);
  t->modifies(lb->rho_micro_CCLabel);
  t->modifies(lb->rho_CCLabel);
  t->modifies(lb->specific_heatLabel);
  t->modifies(lb->gammaLabel);
  t->modifies(lb->thermalCondLabel);
  t->modifies(lb->temp_CCLabel);
  t->modifies(lb->viscosityLabel);
  
  t->computes(d_scalar->scalar_CCLabel);
  t->computes(cumulativeEnergyReleased_CCLabel);
  t->computes(cumulativeEnergyReleased_src_CCLabel);
  if(d_useVariance){
    t->computes(d_scalar->varianceLabel);
  }
  
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
  cout_doing << "Doing Initialize \t\t\t\t\tADIABATIC_TABLE" << endl;
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    int indx = d_matl->getDWIndex();
    
    CCVariable<double>  f, eReleased;
    CCVariable<double> cv, gamma, thermalCond, viscosity, rho_CC, sp_vol;
    CCVariable<double> rho_micro, temp;
    new_dw->allocateAndPut(f, d_scalar->scalar_CCLabel, indx, patch);
    new_dw->allocateAndPut(eReleased, cumulativeEnergyReleased_CCLabel, 
                                                              indx, patch);
    new_dw->getModifiable(cv,          lb->specific_heatLabel,indx,patch);
    new_dw->getModifiable(gamma,       lb->gammaLabel,        indx,patch);
    new_dw->getModifiable(thermalCond, lb->thermalCondLabel,  indx,patch);
    new_dw->getModifiable(viscosity,   lb->viscosityLabel,    indx,patch);
    new_dw->getModifiable(rho_CC,      lb->rho_CCLabel,       indx,patch);
    new_dw->getModifiable(sp_vol,      lb->sp_vol_CCLabel,    indx,patch);
    new_dw->getModifiable(rho_micro,   lb->rho_micro_CCLabel, indx,patch);
    new_dw->getModifiable(temp, lb->temp_CCLabel, indx, patch);
    //__________________________________
    //  initialize the scalar field in a region
    f.initialize(0);

    for(vector<Region*>::iterator iter = d_scalar->regions.begin();
                                  iter != d_scalar->regions.end(); iter++){
      Region* region = *iter;
      for(CellIterator iter = patch->getCellIterator();
          !iter.done(); iter++){
        IntVector c = *iter;
        Point p = patch->cellPosition(c);            
        if(region->piece->inside(p)) {
          f[c] = region->initialScalar;
        }
      } // Over cells
    } // regions
    
    
    if(d_scalar->d_doTableTest){   // 1D table test problem
      for(CellIterator iter = patch->getExtraCellIterator();
            !iter.done(); iter++){
        IntVector c = *iter;
        Point p = patch->cellPosition(c); 
        f[c] = p.x();
      }
    }
    
    setBC(f,"scalar-f", patch, d_sharedState,indx, new_dw); 

    //__________________________________
    // initialize other properties
    vector<constCCVariable<double> > ind_vars;
    ind_vars.push_back(f);
    if(d_useVariance){
      // Variance is zero for initialization
      CCVariable<double> variance;
      new_dw->allocateAndPut(variance, d_scalar->varianceLabel, indx, patch);
      variance.initialize(0);
      ind_vars.push_back(variance);
    }

    // Save the volume fraction so that we can back out rho later
    CCVariable<double> volfrac;
    new_dw->allocateTemporary(volfrac, patch);
    for(CellIterator iter = patch->getExtraCellIterator();!iter.done(); iter++){
      const IntVector& c = *iter;
      volfrac[c] = rho_CC[c]/rho_micro[c];
    }
    
    CellIterator iter = patch->getExtraCellIterator();
    table->interpolate(d_density_index,     rho_micro,   iter, ind_vars);
    table->interpolate(d_gamma_index,       gamma,       iter, ind_vars);
    table->interpolate(d_cv_index,          cv,          iter, ind_vars);
    table->interpolate(d_viscosity_index,   viscosity,   iter, ind_vars);
    table->interpolate(d_temp_index,        temp,        iter, ind_vars);
    table->interpolate(d_thermalcond_index, thermalCond, iter, ind_vars);
    
    CCVariable<double> ref_temp, ref_cv, ref_gamma;
    new_dw->allocateTemporary(ref_cv,    patch);
    new_dw->allocateTemporary(ref_gamma, patch);
    new_dw->allocateTemporary(ref_temp,  patch);
    
    table->interpolate(d_ref_cv_index,    ref_cv,   iter, ind_vars);
    table->interpolate(d_ref_gamma_index, ref_gamma,iter, ind_vars);
    table->interpolate(d_ref_temp_index,  ref_temp, iter, ind_vars);
    
    for(CellIterator iter = patch->getExtraCellIterator();!iter.done();iter++){
      const IntVector& c = *iter;
      // Restore the density and specific volume using the same volume
      // fractions that came from the ICE initialization process
      rho_CC[c] = rho_micro[c]*volfrac[c];
      sp_vol[c] = 1./rho_micro[c];

      double cp  = gamma[c] * cv[c];    
      double icp = ref_gamma[c] * ref_cv[c];
      // This is stupid, but since some machines can't do math when
      // optimization is turned on we trap the common case to make
      // the result be truly zero.  It just makes debugging a bit
      // easier later on.
      if(temp[c] == ref_temp[c] && cp == icp)
        eReleased[c] = 0;
      else
        eReleased[c] = temp[c] * cp - ref_temp[c] * icp;
    }
    setBC(eReleased,"cumulativeEnergyReleased", patch, d_sharedState,indx, new_dw); 

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
void AdiabaticTable::scheduleModifyThermoTransportProperties(SchedulerP& sched,
                                                   const LevelP& level,
                                                   const MaterialSet* /*ice_matls*/)
{
  cout_doing << "ADIABATIC_TABLE::scheduleModifyThermoTransportProperties" << endl;

  Task* t = scinew Task("AdiabaticTable::modifyThermoTransportProperties", 
                   this,&AdiabaticTable::modifyThermoTransportProperties);
                   
  t->requires(Task::OldDW, d_scalar->scalar_CCLabel, Ghost::None,0);  
  t->modifies(lb->specific_heatLabel);
  t->modifies(lb->gammaLabel);
  t->modifies(lb->thermalCondLabel);
  t->modifies(lb->viscosityLabel);
  if(d_useVariance){
    t->requires(Task::NewDW, d_scalar->varianceLabel, Ghost::None, 0);
    t->computes(d_scalar->scaledVarianceLabel);
  }
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
    cout_doing << "Doing modifyThermoTransportProperties on patch "<<patch->getID()
               << "\t ADIABATIC_TABLE" << endl;
   
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
    if(d_useVariance){
      computeScaledVariance(patch, new_dw, indx, f_old, ind_vars);
    }
    
    CellIterator iter = patch->getExtraCellIterator();
    table->interpolate(d_gamma_index,     gamma,    iter, ind_vars);
    table->interpolate(d_cv_index,        cv,       iter, ind_vars);
    table->interpolate(d_viscosity_index, viscosity,iter, ind_vars);
    table->interpolate(d_thermalcond_index, thermalCond, iter, ind_vars);
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
  cout_doing << "Doing computeSpecificHeat on patch "<<patch->getID()
             << "\t ADIABATIC_TABLE" << endl;

  int test_indx = d_matl->getDWIndex();
  //__________________________________
  //  Compute cv for only one matl.
  if (test_indx != indx)
    return;

  constCCVariable<double> f;
  new_dw->get(f,  d_scalar->scalar_CCLabel,  indx, patch, Ghost::None,0);
  
  // interpolate cv
  vector<constCCVariable<double> > ind_vars;
  ind_vars.push_back(f);
  if(d_useVariance){
    computeScaledVariance(patch, new_dw, indx, f, ind_vars);
  }
  
  // Hit the extra cells only on the coarsest level
  // All of the data in the extaCells should be handled by
  // refine_CFI tasks
  CellIterator iterator = patch->getExtraCellIterator();
  int levelIndex = patch->getLevel()->getIndex();
  if(d_doAMR && levelIndex != 0){
    iterator = patch->getCellIterator();
  }
  
  table->interpolate(d_cv_index, cv_new, iterator,ind_vars);
} 

//______________________________________________________________________
void AdiabaticTable::scheduleComputeModelSources(SchedulerP& sched,
                                                 const LevelP& level,
                                                 const ModelInfo* mi)
{
  cout_doing << "ADIABATIC_TABLE::scheduleComputeModelSources " << endl;
  Task* t = scinew Task("AdiabaticTable::computeModelSources", 
                   this,&AdiabaticTable::computeModelSources, mi);
                    
  Ghost::GhostType  gn = Ghost::None;  
  Ghost::GhostType  gac = Ghost::AroundCells;
  t->requires(Task::OldDW, mi->delT_Label,level.get_rep()); 
 
  //t->requires(Task::NewDW, d_scalar->diffusionCoefLabel, gac,1);
  t->requires(Task::OldDW, d_scalar->scalar_CCLabel, gac,1); 
  t->requires(Task::OldDW, mi->rho_CCLabel,          gn);
  t->requires(Task::OldDW, mi->temp_CCLabel,         gn);
  t->requires(Task::OldDW, cumulativeEnergyReleased_CCLabel, gn);

  t->requires(Task::NewDW, lb->specific_heatLabel,   gn);
  t->requires(Task::NewDW, lb->gammaLabel,           gn);

  t->modifies(mi->modelEng_srcLabel);
  t->modifies(cumulativeEnergyReleased_src_CCLabel);
  if(d_scalar->scalar_src_CCLabel){
    t->modifies(d_scalar->scalar_src_CCLabel);
  }
  if(d_useVariance){
    t->requires(Task::NewDW, d_scalar->varianceLabel, gn, 0);
  }

  // Interpolated table values
  for(int i=0;i<(int)tablevalues.size();i++){
    TableValue* tv = tablevalues[i];
    t->computes(tv->label);
  }
  sched->addTask(t, level->eachPatch(), d_matl_set);
}

//______________________________________________________________________
void AdiabaticTable::computeModelSources(const ProcessorGroup*, 
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
    cout_doing << "Doing momentumAndEnergyExch... on patch "<<patch->getID()
               << "\t\tADIABATIC_TABLE" << endl;

    for(int m=0;m<matls->size();m++){
      int matl = matls->get(m);

      // Get mixture fraction, and initialize source to zero
      constCCVariable<double> f_old, rho_CC, oldTemp, cv, gamma;
      constCCVariable<double> eReleased;
      CCVariable<double> energySource, sp_vol_ref, flameTemp;
      CCVariable<double> eReleased_src;

      old_dw->get(f_old,    d_scalar->scalar_CCLabel,   matl, patch, gn, 0);
      old_dw->get(rho_CC,   mi->rho_CCLabel,            matl, patch, gn, 0);
      old_dw->get(oldTemp,  mi->temp_CCLabel,           matl, patch, gn, 0);
      old_dw->get(eReleased, cumulativeEnergyReleased_CCLabel,
                                                        matl, patch, gn, 0);
      new_dw->get(cv,       lb->specific_heatLabel,     matl, patch, gn, 0);
      new_dw->get(gamma,    lb->gammaLabel,             matl, patch, gn, 0);
      new_dw->getModifiable(energySource,   mi->modelEng_srcLabel,  
                                                        matl, patch);
      new_dw->getModifiable(eReleased_src,  cumulativeEnergyReleased_src_CCLabel,
                            matl, patch);
      new_dw->allocateTemporary(flameTemp, patch); 
      new_dw->allocateTemporary(sp_vol_ref, patch);
      
      //__________________________________
      //  grab values from the tables
      vector<constCCVariable<double> > ind_vars;
      ind_vars.push_back(f_old);
      if(d_useVariance){
        computeScaledVariance(patch, new_dw, matl, f_old, ind_vars);
      }

      CellIterator iter = patch->getExtraCellIterator();
      
      table->interpolate(d_temp_index,      flameTemp,  iter, ind_vars);
      CCVariable<double> ref_temp;
      new_dw->allocateTemporary(ref_temp, patch); 
      table->interpolate(d_ref_temp_index, ref_temp, iter, ind_vars);
      CCVariable<double> ref_cv;
      new_dw->allocateTemporary(ref_cv, patch); 
      table->interpolate(d_ref_cv_index,    ref_cv, iter, ind_vars);
      CCVariable<double> ref_gamma;
      new_dw->allocateTemporary(ref_gamma, patch); 
      table->interpolate(d_ref_gamma_index, ref_gamma, iter, ind_vars);

      Vector dx = patch->dCell();
      double volume = dx.x()*dx.y()*dx.z();
      
      
      for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
        IntVector c = *iter;

        double mass      = rho_CC[c]*volume;
        // cp might need to change to be interpolated from the table
        // when we make the cp in the dw correct
        double cp        = gamma[c] * cv[c];
        double ref_cp    = ref_gamma[c] * ref_cv[c];
        double energyNew = flameTemp[c] * cp;
        
        double energyOrig = ref_temp[c] * ref_cp;
        double erelease   = (energyNew-energyOrig) - eReleased[c];

        // Add energy released to the cumulative total and hand it to ICE
        // as a source
        eReleased_src[c] += erelease;
        energySource[c] += erelease*mass;
      }
      
      //__________________________________
      //  table sanity test
      if(d_scalar->d_doTableTest){
        CCVariable<double> rho_table, mix_mol_weight, co2, h2o;
        new_dw->allocateTemporary(rho_table, patch);
        new_dw->allocateTemporary(co2, patch);
        new_dw->allocateTemporary(h2o, patch);
        table->interpolate(d_density_index, rho_table,     iter, ind_vars);
        
        // get co2 and h2o
        for(int i=0;i<(int)tablevalues.size();i++){
          TableValue* tv = tablevalues[i];
          CellIterator iter = patch->getCellIterator();
          if(tv->name == "CO2"){
            table->interpolate(tv->index, co2, iter, ind_vars);
          }
          if(tv->name == "H2O"){
            table->interpolate(tv->index, h2o, iter, ind_vars);
          }
        }
        
        cout.setf(ios::scientific,ios::floatfield);
        cout.precision(10);
    
        cout << "                 MixtureFraction,                      temp_table,       gamma,             cv,             rho_table,      press_thermo,   (gamma-1)cv,   rho_table*temp_table,  co2,           h2o"<< endl;        
        for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
          IntVector c = *iter;
          double press = (rho_table[c] * cv[c] * (gamma[c]-1) * flameTemp[c]);
          double thermo = cv[c] * (gamma[c]-1);
          double physical = rho_table[c] * flameTemp[c];
          cout << level->getCellPosition(c) << " " << flameTemp[c] <<  " " << gamma[c] << " " << cv[c] << " "  << " " << rho_table[c] << " " << press << " " <<thermo << " " <<physical <<" " <<co2[c] << " " << h2o[c] <<endl;
        }
        cout.setf(ios::scientific ,ios::floatfield);
      }

      //__________________________________
      //  Tack on diffusion
      double diff_coeff = d_scalar->diff_coeff;
      if(diff_coeff != 0.0){ 
        CCVariable<double> f_src;
        new_dw->getModifiable(f_src,  d_scalar->scalar_src_CCLabel,
                              matl, patch);
        
        bool use_vol_frac = false; // don't include vol_frac in diffusion calc.
        CCVariable<double> placeHolder;
        
        CCVariable<double> diff_coeff_cc;
        new_dw->allocateTemporary(diff_coeff_cc, patch);
        diff_coeff_cc.initialize(diff_coeff);
        
        scalarDiffusionOperator(new_dw, patch, use_vol_frac, f_old,
                                placeHolder, f_src, diff_coeff_cc, delT);
      }

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
      
      // Compute miscellaneous table quantities.  Users can request
      // arbitrary quanitities to be interplated by adding an entry
      // of the form: <tableValue>N2</tableValue> to the model section.
      // This is useful for computing derived quantities such as species
      // concentrations.
      for(int i=0;i<(int)tablevalues.size();i++){
        TableValue* tv = tablevalues[i];
        CCVariable<double> value;
        new_dw->allocateAndPut(value, tv->label, matl, patch);
        CellIterator iter = patch->getExtraCellIterator();
        table->interpolate(tv->index, value, iter, ind_vars);
        if(patch->getID() == 0){ 
          cerr << "interpolating " << tv->name << '\n';
        }
      }
    }
  }
}

//______________________________________________________________________
void AdiabaticTable::computeScaledVariance(const Patch* patch,
                                           DataWarehouse* new_dw,
                                           const int indx,
                                           constCCVariable<double> f,
                                           vector<constCCVariable<double> >& ind_vars)
{
  CCVariable<double> scaledvariance;
  constCCVariable<double> variance;
  new_dw->get(variance, d_scalar->varianceLabel, indx, patch, Ghost::None, 0);
  new_dw->allocateAndPut(scaledvariance, d_scalar->scaledVarianceLabel,
                         indx, patch);
                         
  for(CellIterator iter = patch->getExtraCellIterator(); !iter.done(); iter++){
    const IntVector& c = *iter;
    double denom = f[c] * (1-f[c]);
    if(denom < 1.e-20){
      denom = 1.e-20;
    }
    double sv = variance[c] / denom;
    
    //__________________________________
    //  clamp the  scaled variance
    if(sv < 0){
      sv = 0;
    }else if(sv > d_varianceMax){
      sv = d_varianceMax;
    }
    scaledvariance[c] = sv;
  }
  ind_vars.push_back(scaledvariance);
}

//______________________________________________________________________
void AdiabaticTable::scheduleTestConservation(SchedulerP& sched,
                                              const PatchSet* patches,
                                              const ModelInfo* mi)
{
  if(d_scalar->d_test_conservation){
    cout_doing << "ADIABATICTABLE::scheduleTestConservation " << endl;
    Task* t = scinew Task("AdiabaticTable::testConservation", 
                     this,&AdiabaticTable::testConservation, mi);

    t->requires(Task::OldDW, mi->delT_Label,getLevel(patches)); 
    Ghost::GhostType  gn = Ghost::None;
    // compute sum(scalar_f * mass)
    t->requires(Task::NewDW, d_scalar->scalar_CCLabel, gn,0); 
    t->requires(Task::NewDW, mi->rho_CCLabel,          gn,0);
    t->requires(Task::NewDW, lb->uvel_FCMELabel,       gn,0); 
    t->requires(Task::NewDW, lb->vvel_FCMELabel,       gn,0); 
    t->requires(Task::NewDW, lb->wvel_FCMELabel,       gn,0); 
    t->computes(d_scalar->sum_scalar_fLabel);

    sched->addTask(t, patches, d_matl_set);
  }
}

//______________________________________________________________________
void AdiabaticTable::testConservation(const ProcessorGroup*, 
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
               << "\t\t\t AdiabaticTable" << endl;
               
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
    
    new_dw->put(sum_vartype(sum_mass_f), d_scalar->sum_scalar_fLabel);
  }
}
//__________________________________      
void AdiabaticTable::scheduleComputeStableTimestep(SchedulerP&,
                                      const LevelP&,
                                      const ModelInfo*)
{
  // None necessary...
}


//______________________________________________________________________
//
void AdiabaticTable::scheduleErrorEstimate(const LevelP& coarseLevel,
                                          SchedulerP& sched)
{
  cout_doing << "AdiabaticTable::scheduleErrorEstimate \t\t\tL-" 
             << coarseLevel->getIndex() << '\n';
  
  Task* t = scinew Task("AdiabaticTable::errorEstimate", 
                  this, &AdiabaticTable::errorEstimate, false);  
  
  Ghost::GhostType  gac  = Ghost::AroundCells; 
  t->requires(Task::NewDW, d_scalar->scalar_CCLabel,  gac, 1);
  
  t->computes(d_scalar->mag_grad_scalarLabel);
  t->modifies(d_sharedState->get_refineFlag_label(),      d_sharedState->refineFlagMaterials(), Task::OutOfDomain);
  t->modifies(d_sharedState->get_refinePatchFlag_label(), d_sharedState->refineFlagMaterials(), Task::OutOfDomain);
  
  sched->addTask(t, coarseLevel->eachPatch(), d_sharedState->allICEMaterials());
}
/*_____________________________________________________________________
 Function~  AdiabaticTable::errorEstimate--
______________________________________________________________________*/
void AdiabaticTable::errorEstimate(const ProcessorGroup*,
                                   const PatchSubset* patches,
                                   const MaterialSubset*,
                                   DataWarehouse*,
                                   DataWarehouse* new_dw,
                                   bool)
{
  cout_doing << "Doing errorEstimate \t\t\t\t\t AdiabaticTable"<< endl;
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    
    Ghost::GhostType  gac  = Ghost::AroundCells;
    const VarLabel* refineFlagLabel = d_sharedState->get_refineFlag_label();
    const VarLabel* refinePatchLabel= d_sharedState->get_refinePatchFlag_label();
    
    CCVariable<int> refineFlag;
    new_dw->getModifiable(refineFlag, refineFlagLabel, 0, patch);      

    PerPatch<PatchFlagP> refinePatchFlag;
    new_dw->get(refinePatchFlag, refinePatchLabel, 0, patch);

    int indx = d_matl->getDWIndex();
    constCCVariable<double> f;
    CCVariable<double> mag_grad_f;
    
    new_dw->get(f,                     d_scalar->scalar_CCLabel,  indx ,patch,gac,1);
    new_dw->allocateAndPut(mag_grad_f, d_scalar->mag_grad_scalarLabel, 
                           indx,patch);
    mag_grad_f.initialize(0.0);
    
    //__________________________________
    // compute gradient
    Vector dx = patch->dCell(); 
    
    for(CellIterator iter = patch->getCellIterator();!iter.done();iter++){
      IntVector c = *iter;
      
      Vector grad_f;
      for(int dir = 0; dir <3; dir ++ ) { 
        IntVector r = c;
        IntVector l = c;
        double inv_dx = 0.5 /dx[dir];
        r[dir] += 1;
        l[dir] -= 1;
        grad_f[dir] = (f[r] - f[l])*inv_dx;
      }
      mag_grad_f[c] = grad_f.length();
    }
    //__________________________________
    // set refinement flag
    PatchFlag* refinePatch = refinePatchFlag.get().get_rep();
    for(CellIterator iter = patch->getCellIterator();!iter.done();iter++){
      IntVector c = *iter;
      if( mag_grad_f[c] > d_scalar->refineCriteria){
        refineFlag[c] = true;
        refinePatch->set();
      }
    }
  }  // patches
}
