/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
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


// TODO
// Use cp directly instead of cv/gamma
// multimaterial checks in smagorisky
// multimaterial checks in dynamic model
// Avoid recomputation of variance

#include <CCA/Components/ICE/ICEMaterial.h>
#include <CCA/Components/ICE/ConservationTest.h>
#include <CCA/Components/ICE/BoundaryCond.h>
#include <CCA/Components/ICE/Diffusion.h>
#include <CCA/Components/Models/FluidsBased/NonAdiabaticTable.h>
#include <CCA/Components/Models/FluidsBased/TableFactory.h>
#include <CCA/Components/Models/FluidsBased/TableInterface.h>
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
NonAdiabaticTable::NonAdiabaticTable(const ProcessorGroup* myworld, 
                     ProblemSpecP& params)
  : ModelInterface(myworld), params(params)
{
  d_scalar = 0;
  d_matl_set = 0;
  lb  = scinew ICELabel();
  cumulativeEnergyReleased_CCLabel = VarLabel::create("cumulativeEnergyReleased", CCVariable<double>::getTypeDescription());
  cumulativeEnergyReleased_src_CCLabel = VarLabel::create("cumulativeEnergyReleased_src", CCVariable<double>::getTypeDescription());
}


//__________________________________
NonAdiabaticTable::~NonAdiabaticTable()
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
    if(d_scalar->scalar_src_CCLabel){
      VarLabel::destroy(d_scalar->scalar_src_CCLabel);
    }
  }
  VarLabel::destroy(cumulativeEnergyReleased_CCLabel);
  VarLabel::destroy(cumulativeEnergyReleased_src_CCLabel);
  
  delete lb;
  delete table;
  for(int i=0;i<(int)tablevalues.size();i++)
    delete tablevalues[i];
  
  for(vector<Region*>::iterator iter = d_scalar->regions.begin();
                                iter != d_scalar->regions.end(); iter++){
    Region* region = *iter;
    delete region;

  }
}

//__________________________________
NonAdiabaticTable::Region::Region(GeometryPieceP piece, ProblemSpecP& ps)
  : piece(piece)
{
  ps->require("scalar", initialScalar);
}
//______________________________________________________________________
//     P R O B L E M   S E T U P
void NonAdiabaticTable::problemSetup(GridP&, SimulationStateP& in_state,
                        ModelSetup* setup)
{
  cout_doing << "Doing problemSetup \t\t\t\tADIABATIC_TABLE" << endl;
  sharedState = in_state;
  d_matl = sharedState->parseAndLookupMaterial(params, "material");

  vector<int> m(1);
  m[0] = d_matl->getDWIndex();
  d_matl_set = scinew MaterialSet();
  d_matl_set->addAll(m);
  d_matl_set->addReference();

  // Get parameters
  params->getWithDefault("varianceScale", varianceScale, 0.0);
  params->getWithDefault("varianceMax", varianceMax, 1.0);
  useVariance = (varianceScale != 0.0);

  //__________________________________
  //setup the table
  string tablename = "adiabatic";
  table = TableFactory::readTable(params, tablename);
  table->addIndependentVariable("F");
  if(useVariance)
    table->addIndependentVariable("Fvar");
  
  for (ProblemSpecP child = params->findBlock("tableValue"); child != 0;
       child = child->findNextBlock("tableValue")) {
    TableValue* tv = scinew TableValue;
    child->get(tv->name);
    tv->index = table->addDependentVariable(tv->name);
    string labelname = tv->name;
    tv->label = VarLabel::create(labelname, CCVariable<double>::getTypeDescription());
    tablevalues.push_back(tv);
  }
  
  d_temp_index          = table->addDependentVariable("Temp");
  d_density_index       = table->addDependentVariable("density");
  d_gamma_index         = table->addDependentVariable("gamma");
  d_cv_index            = table->addDependentVariable("heat_capac_Cv");
  d_viscosity_index     = table->addDependentVariable("viscosity");
  d_thermalcond_index   = table->addDependentVariable("thermal_conductivity");
  d_ref_cv_index    = table->addDependentVariable("reference_heat_capac_Cv");
  d_ref_gamma_index = table->addDependentVariable("reference_gamma");
  d_ref_temp_index  = table->addDependentVariable("reference_Temp");
  
  bool cerrSwitch = (d_myworld->myrank() == 0); 
  table->setup(cerrSwitch);

#if 0
  ofstream out("graph.dat");
  int ng = 100;
  vector<double> mm;
  int nv;
  if(useVariance){
    mm.resize(2);
    nv = 10;
  } else {
    mm.resize(1);
    nv = 0;
  }
  for(int j=0;j<=nv;j++){
    if(useVariance)
      mm[1] = j/(double)nv;
    for(int i=0;i<=ng;i++){
      double mix = i/(double)ng;
      mm[0]=mix;
      double pressure = table->interpolate(d_temp_index, mm) 
        * table->interpolate(d_cv_index, mm)
        * (table->interpolate(d_gamma_index, mm)-1)
        * table->interpolate(d_density_index, mm);
      out << mix << " "
          << table->interpolate(d_temp_index, mm) << " "
          << pressure << " "
          << table->interpolate(d_density_index, mm) << " "
          << table->interpolate(d_gamma_index, mm) << " "
          << table->interpolate(d_cv_index, mm) << " "
          << table->interpolate(d_viscosity_index, mm) << " "
          << table->interpolate(d_thermalcond_index, mm) << " "
          << table->interpolate(d_ref_cv_index, mm) << " "
          << table->interpolate(d_ref_gamma_index, mm) << " "
          << table->interpolate(d_ref_temp_index, mm) << "\n";
    }
    out << '\n';
  }
#endif

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
     throw ProblemSetupException("NonAdiabaticTable: Couldn't find scalar tag", __FILE__, __LINE__);    
   }
   
   child->getWithDefault("test_conservation", d_test_conservation, false);
   
   ProblemSpecP const_ps = child->findBlock("constants");
   if(!const_ps) {
     throw ProblemSetupException("NonAdiabaticTable:Couldn't find constants tag", __FILE__, __LINE__);
   }
 
   const_ps->getWithDefault("diffusivity",  d_scalar->diff_coeff, 0.0);
   // scalar src is only needed if we have non-zero diffusivity
   if(d_scalar->diff_coeff)
     d_scalar->scalar_src_CCLabel = VarLabel::create("scalar-f_src",
                                                     td_CCdouble);
   else
     d_scalar->scalar_src_CCLabel = 0;

   // Tell ICE to transport the scalar and the energy
   setup->registerTransportedVariable(d_matl_set,
                                      d_scalar->scalar_CCLabel,
                                      d_scalar->scalar_src_CCLabel);
   setup->registerTransportedVariable(d_matl_set,
                                      cumulativeEnergyReleased_CCLabel,
                                      cumulativeEnergyReleased_src_CCLabel);
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
void NonAdiabaticTable::scheduleInitialize(SchedulerP& sched,
                                   const LevelP& level,
                                   const ModelInfo*)
{
  cout_doing << "ADIABATIC_TABLE::scheduleInitialize " << endl;
  Task* t = scinew Task("NonAdiabaticTable::initialize", this, 
                        &NonAdiabaticTable::initialize);

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
  
  sched->addTask(t, level->eachPatch(), d_matl_set);
}
//______________________________________________________________________
//       I N I T I A L I Z E
void NonAdiabaticTable::initialize(const ProcessorGroup*, 
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
    
    setBC(f,"scalar-f", patch, sharedState,indx, new_dw); 

    //__________________________________
    // initialize other properties
    vector<constCCVariable<double> > ind_vars;
    ind_vars.push_back(f);
    if(useVariance){
      // Variance is zero for initialization
      CCVariable<double> variance;
      new_dw->allocateTemporary(variance, patch);
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
    setBC(eReleased,"cumulativeEnergyReleased", patch, sharedState,indx, new_dw); 

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
void NonAdiabaticTable::scheduleModifyThermoTransportProperties(SchedulerP& sched,
                                                   const LevelP& level,
                                                   const MaterialSet* /*ice_matls*/)
{
  cout_doing << "ADIABATIC_TABLE::scheduleModifyThermoTransportProperties" << endl;

  Task* t = scinew Task("NonAdiabaticTable::modifyThermoTransportProperties", 
                   this,&NonAdiabaticTable::modifyThermoTransportProperties);
                   
  t->requires(Task::OldDW, d_scalar->scalar_CCLabel, Ghost::None,0);  
  t->modifies(lb->specific_heatLabel);
  t->modifies(lb->gammaLabel);
  t->modifies(lb->thermalCondLabel);
  t->modifies(lb->viscosityLabel);
  if(useVariance){
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
void NonAdiabaticTable::modifyThermoTransportProperties(const ProcessorGroup*, 
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
    CCVariable<double> scaledvariance;
    if(useVariance){
      constCCVariable<double> variance;
      new_dw->get(variance, d_scalar->varianceLabel, indx, patch,
                  Ghost::None, 0);
      
      // This is put into the DW instead of allocated as a temporary
      // so that we can save it.
      new_dw->allocateAndPut(scaledvariance, d_scalar->scaledVarianceLabel,
                             indx, patch);
      for(CellIterator iter = patch->getExtraCellIterator(); !iter.done(); iter++){
        const IntVector& c = *iter;
        double denom = f_old[c] * (1-f_old[c]);
        if(denom < 1.e-20)
          denom = 1.e-20;
        double sv = variance[c] / denom;
        if(sv < 0)
          sv = 0;
        else if(sv > varianceMax)
          sv = varianceMax;
        scaledvariance[c] = sv;
      }
      ind_vars.push_back(scaledvariance);
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
void NonAdiabaticTable::computeSpecificHeat(CCVariable<double>& cv_new,
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
  CCVariable<double> scaledvariance;
  if(useVariance){
    constCCVariable<double> variance;
    new_dw->get(variance, d_scalar->varianceLabel, indx, patch, Ghost::None, 0);

    new_dw->allocateTemporary(scaledvariance, patch);
    for(CellIterator iter = patch->getExtraCellIterator(); !iter.done(); iter++){
      const IntVector& c = *iter;
      double denom = f[c] * (1-f[c]);
      if(denom < 1.e-20)
        denom = 1.e-20;
      double sv = variance[c] / denom;
      if(sv < 0)
        sv = 0;
      else if(sv > varianceMax)
        sv = varianceMax;
      scaledvariance[c] = sv;
    }
        
    ind_vars.push_back(scaledvariance);
  }
  table->interpolate(d_cv_index, cv_new, patch->getExtraCellIterator(),
                     ind_vars);
} 

//______________________________________________________________________
void NonAdiabaticTable::scheduleComputeModelSources(SchedulerP& sched,
                                                 const LevelP& level,
                                                 const ModelInfo* mi)
{
  cout_doing << "ADIABATIC_TABLE::scheduleComputeModelSources " << endl;
  Task* t = scinew Task("NonAdiabaticTable::computeModelSources", 
                   this,&NonAdiabaticTable::computeModelSources, mi);
                    
  Ghost::GhostType  gn = Ghost::None;  
  Ghost::GhostType  gac = Ghost::AroundCells;
 
  //t->requires(Task::NewDW, d_scalar->diffusionCoefLabel, gac,1);
  t->requires(Task::OldDW, mi->delT_Label,           level.get_rep());
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
  if(useVariance){
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
void NonAdiabaticTable::computeModelSources(const ProcessorGroup*, 
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
      CCVariable<double> scaledvariance;
      if(useVariance){
        constCCVariable<double> variance;
        new_dw->get(variance, d_scalar->varianceLabel, matl, patch, gn, 0);

        new_dw->allocateTemporary(scaledvariance, patch);
        for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
          const IntVector& c = *iter;
          double denom = f_old[c] * (1-f_old[c]);
          if(denom < 1.e-20)
            denom = 1.e-20;
          double sv = variance[c] / denom;
          if(sv < 0)
            sv = 0;
          else if(sv > varianceMax)
            sv = varianceMax;
          scaledvariance[c] = sv;
        }
        
        ind_vars.push_back(scaledvariance);
      }

      CellIterator iter = patch->getCellIterator();
      table->interpolate(d_temp_index,  flameTemp,  iter, ind_vars);

      CCVariable<double> ref_temp;
      new_dw->allocateTemporary(ref_temp, patch); 
      table->interpolate(d_ref_temp_index, ref_temp, iter, ind_vars);
      CCVariable<double> ref_cv;
      new_dw->allocateTemporary(ref_cv, patch); 
      table->interpolate(d_ref_cv_index, ref_cv, iter, ind_vars);
      CCVariable<double> ref_gamma;
      new_dw->allocateTemporary(ref_gamma, patch); 
      table->interpolate(d_ref_gamma_index, ref_gamma, iter, ind_vars);

      Vector dx = patch->dCell();
      double volume = dx.x()*dx.y()*dx.z();
      
#if 0
      double maxTemp = 0;
      double maxIncrease = 0;    //debugging
      double maxDecrease = 0;
      double totalEnergy = 0;
      double maxFlameTemp=0;
      double esum = 0;
      double fsum = 0;
      int ncells = 0;
      double cpsum = 0;
      double masssum=0;
#endif

      for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
        IntVector c = *iter;

        double mass      = rho_CC[c]*volume;
        // cp might need to change to be interpolated from the table
        // when we make the cp in the dw correct
        double cp        = gamma[c] * cv[c];
        double energyNew = flameTemp[c] * cp;
        // double icp = initial_cp[c];
        double ref_cp = ref_gamma[c] * ref_cv[c];
        double energyOrig = ref_temp[c] * ref_cp;
        double erelease = (energyNew-energyOrig) - eReleased[c];

        // Add energy released to the cumulative total and hand it to ICE
        // as a source
        eReleased_src[c] += erelease;
        energySource[c] += erelease*mass;

#if 0
        //__________________________________
        // debugging
        totalEnergy += erelease*mass;
        fsum += f_old[c] * mass;
        esum += oldTemp[c]*cp*mass;
        cpsum += cp;
        masssum += mass;
        ncells++;
        double newTemp = oldTemp[c] + erelease/cp;
        if(newTemp > maxTemp)
          maxTemp = newTemp;
        if(flameTemp[c] > maxFlameTemp)
          maxFlameTemp = flameTemp[c];
        double dtemp = newTemp-oldTemp[c];
        if(dtemp > maxIncrease)
          maxIncrease = dtemp;
        if(dtemp < maxDecrease)
          maxDecrease = dtemp;
#endif
      }
#if 0
      cerr << "MaxTemp = " << maxTemp << ", maxFlameTemp=" << maxFlameTemp << ", maxIncrease=" << maxIncrease << ", maxDecrease=" << maxDecrease << ", totalEnergy=" << totalEnergy << '\n';
      double cp = cpsum/ncells;
      double mass = masssum/ncells;
      double e = esum/ncells;
      double atemp = e/(mass*cp);
      vector<double> tmp(1);
      tmp[0]=fsum/masssum;
      cerr << "AverageTemp=" << atemp << ", AverageF=" << fsum/masssum << ", targetTemp=" << table->interpolate(d_temp_index, tmp) << '\n';
#endif

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
        cerr << "interpolating " << tv->name << '\n';
        CCVariable<double> value;
        new_dw->allocateAndPut(value, tv->label, matl, patch);
        CellIterator iter = patch->getCellIterator();
        table->interpolate(tv->index, value, iter, ind_vars);
      }
    }
  }
}
//______________________________________________________________________
void NonAdiabaticTable::scheduleTestConservation(SchedulerP& sched,
                                              const PatchSet* patches,
                                              const ModelInfo* mi)
{
  if(d_test_conservation){
    cout_doing << "ADIABATICTABLE::scheduleTestConservation " << endl;
    Task* t = scinew Task("NonAdiabaticTable::testConservation", 
                     this,&NonAdiabaticTable::testConservation, mi);

    Ghost::GhostType  gn = Ghost::None;
    // compute sum(scalar_f * mass)
    t->requires(Task::OldDW, mi->delT_Label, getLevel(patches));
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
void NonAdiabaticTable::testConservation(const ProcessorGroup*, 
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
               << "\t\t\t NonAdiabaticTable" << endl;
               
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
void NonAdiabaticTable::scheduleComputeStableTimestep(SchedulerP&,
                                      const LevelP&,
                                      const ModelInfo*)
{
  // None necessary...
}


//______________________________________________________________________
//
void NonAdiabaticTable::scheduleErrorEstimate(const LevelP&,
                                           SchedulerP&)
{
}
