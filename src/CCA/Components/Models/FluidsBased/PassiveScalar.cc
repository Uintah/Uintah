/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
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


#include <CCA/Components/ICE/Core/ConservationTest.h>
#include <CCA/Components/ICE/Core/Diffusion.h>
#include <CCA/Components/ICE/CustomBCs/BoundaryCond.h>
#include <CCA/Components/Models/FluidsBased/PassiveScalar.h>
#include <CCA/Components/Models/FluidsBased/FluidsBasedModel.h>
#include <Core/Grid/Variables/PerPatchVars.h>
#include <CCA/Ports/Scheduler.h>
#include <CCA/Ports/Regridder.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/GeometryPiece/GeometryPieceFactory.h>
#include <Core/GeometryPiece/UnionGeometryPiece.h>
#include <Core/Grid/Box.h>
#include <Core/Grid/DbgOutput.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Material.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/PerPatch.h>

#include <Core/Grid/MaterialManager.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Exceptions/ParameterNotFound.h>
#include <Core/Util/DOUT.hpp>

#include <iostream>
#include <cstdio>

using namespace Uintah;
using namespace std;


Dout dout_models_ps("MODELS_DOING_COUT", "Models::PassiveScalar", "Models::PassiveScalar debug stream", false);
//______________________________________________________________________
PassiveScalar::PassiveScalar(const ProcessorGroup* myworld, 
                             const MaterialManagerP& materialManager,
                             const ProblemSpecP& params)
  : FluidsBasedModel(myworld, materialManager), d_params(params)
{
  m_modelComputesThermoTransportProps = true;
  
  d_matl_set = 0;
  Ilb  = scinew ICELabel();
  Slb = scinew PassiveScalarLabel();
}

//__________________________________
PassiveScalar::~PassiveScalar()
{
  if(d_matl_set && d_matl_set->removeReference()) {
    delete d_matl_set;
  }
  
  VarLabel::destroy( d_scalar->scalar_CCLabel );
  VarLabel::destroy( d_scalar->scalar_source_CCLabel );
  VarLabel::destroy( d_scalar->diffusionCoefLabel );
  VarLabel::destroy( d_scalar->mag_grad_scalarLabel );
  VarLabel::destroy( Slb->sum_scalar_fLabel );

  delete Ilb;
  delete Slb;
  
  // regions used during initialization
  for(vector<Region*>::iterator iter = d_scalar->regions.begin();
                                iter != d_scalar->regions.end(); iter++){
    Region* region = *iter;
    delete region;
  }
  
  // Interior regions
  for(vector<interiorRegion*>::iterator iter = d_scalar->interiorRegions.begin();
                                        iter != d_scalar->interiorRegions.end(); iter++){
    interiorRegion* region = *iter;
    delete region;
  }
  
}

//______________________________________________________________________
//
PassiveScalar::Region::Region(GeometryPieceP piece, ProblemSpecP& ps)
  : piece(piece)
{
  ps->require("scalar", initialScalar);
  ps->getWithDefault("sinusoidalInitialize",      sinusoidalInitialize,     false);
  ps->getWithDefault("linearInitialize",          linearInitialize,         false);
  ps->getWithDefault("cubicInitialize",           cubicInitialize,          false);
  ps->getWithDefault("quadraticInitialize",       quadraticInitialize,      false);
  ps->getWithDefault("exponentialInitialize_1D",  exponentialInitialize_1D, false);
  ps->getWithDefault("exponentialInitialize_2D",  exponentialInitialize_2D, false);
  ps->getWithDefault("triangularInitialize",      triangularInitialize,     false);
  
  if(sinusoidalInitialize){
    ps->getWithDefault("freq",freq,IntVector(0,0,0));
  }
  if(linearInitialize || triangularInitialize){
    ps->getWithDefault("slope",slope,Vector(0,0,0));
  }
  if(quadraticInitialize || exponentialInitialize_1D || exponentialInitialize_2D){
    ps->getWithDefault("coeff",coeff,Vector(0,0,0));
  }
  
  uniformInitialize = true;
  if(sinusoidalInitialize    || linearInitialize || 
     quadraticInitialize     || cubicInitialize || 
     exponentialInitialize_1D|| exponentialInitialize_2D || 
     triangularInitialize){
    uniformInitialize = false;
  }
}

//______________________________________________________________________
//
PassiveScalar::interiorRegion::interiorRegion(GeometryPieceP piece, ProblemSpecP& ps)
  : piece(piece)
{
  ps->require("scalar", value);
  ps->getWithDefault( "maxScalar" , clampValue, DBL_MAX );
}


//______________________________________________________________________
//     P R O B L E M   S E T U P
void PassiveScalar::problemSetup(GridP&, const bool isRestart)
{
  DOUT(dout_models_ps, "Doing problemSetup \t\t\t\tPASSIVE_SCALAR" );

  ProblemSpecP PS_ps = d_params->findBlock("PassiveScalar");
  d_matl = m_materialManager->parseAndLookupMaterial(PS_ps, "material");

  vector<int> m(1);
  m[0] = d_matl->getDWIndex();
  d_matl_set = scinew MaterialSet();
  d_matl_set->addAll(m);
  d_matl_set->addReference();
  d_matl_sub = d_matl_set->getUnion();
  
  //__________________________________
  // - create Label names
  // - register the scalar to be transported
  d_scalar = scinew Scalar();
  d_scalar->index = 0;
  d_scalar->name  = "f";
  
  const TypeDescription* td_CCdouble = CCVariable<double>::getTypeDescription();
  //const TypeDescription* td_CCVector = CCVariable<Vector>::getTypeDescription();
    
  d_scalar->scalar_CCLabel =        VarLabel::create("scalar-f",         td_CCdouble);
  d_scalar->diffusionCoefLabel =    VarLabel::create("scalar-diffCoef",  td_CCdouble);
  d_scalar->scalar_source_CCLabel = VarLabel::create("scalar-f_src",     td_CCdouble);
  d_scalar->mag_grad_scalarLabel =  VarLabel::create("mag_grad_scalar-f",td_CCdouble);                                 
  
  Slb->sum_scalar_fLabel      =     VarLabel::create("sum_scalar_f",    sum_vartype::getTypeDescription());
  
  registerTransportedVariable(d_matl_set,
                              d_scalar->scalar_CCLabel,
                              d_scalar->scalar_source_CCLabel);  

  //__________________________________
  //  register the AMRrefluxing variables                               
  if(m_AMR){
    registerAMRRefluxVariable(d_matl_set, d_scalar->scalar_CCLabel);
  }
  //__________________________________
  // Read in the constants for the scalar
  ProblemSpecP child = PS_ps->findBlock("scalar");
  if (!child){
    throw ProblemSetupException("PassiveScalar: Couldn't find scalar tag", __FILE__, __LINE__);    
  }

  child->getWithDefault("test_conservation", d_test_conservation, false);

  ProblemSpecP const_ps = child->findBlock("constants");
  if(!const_ps) {
    throw ProblemSetupException("PassiveScalar: Couldn't find constants tag", __FILE__, __LINE__);
  }

  const_ps->getWithDefault("initialize_diffusion_knob",  d_scalar->initialize_diffusion_knob,   0);
  const_ps->getWithDefault("diffusivity",                d_scalar->diff_coeff, 0.0);
  const_ps->getWithDefault("AMR_Refinement_Criteria",    d_scalar->refineCriteria,1e100);

  //__________________________________
  //  Initialization: Read in the geometry objects for the scalar
  ProblemSpecP init_ps = child->findBlock("initialization");
  if( !isRestart ){
   for ( ProblemSpecP geom_obj_ps = init_ps->findBlock("geom_object"); geom_obj_ps != nullptr; geom_obj_ps = geom_obj_ps->findNextBlock("geom_object") ) {
    vector<GeometryPieceP> pieces;
    GeometryPieceFactory::create(geom_obj_ps, pieces);

    GeometryPieceP mainpiece;
    if(pieces.size() == 0){
      throw ParameterNotFound("No piece specified in geom_object", __FILE__, __LINE__);
    }
    else if(pieces.size() > 1){
      mainpiece = scinew UnionGeometryPiece(pieces);
    }
    else {
      mainpiece = pieces[0];
    }

    d_scalar->regions.push_back(scinew Region(mainpiece, geom_obj_ps));
   }
  }
  
  if(d_scalar->regions.size() == 0 && !isRestart) {
    throw ProblemSetupException("Variable: scalar-f does not have any initial value regions", __FILE__, __LINE__);
  }
  
  //__________________________________
  //  Read in interior geometry objects for injecting a scalar in the domain
  ProblemSpecP srcs_ps = child->findBlock("interiorSources");
  if( srcs_ps ) {
    for (ProblemSpecP geom_obj_ps = srcs_ps->findBlock("geom_object"); geom_obj_ps != nullptr; geom_obj_ps = geom_obj_ps->findNextBlock("geom_object") ) {
      vector<GeometryPieceP> pieces;
      GeometryPieceFactory::create(geom_obj_ps, pieces);

      GeometryPieceP mainpiece;
      if(pieces.size() == 0){
        throw ParameterNotFound("No piece specified in geom_object", __FILE__, __LINE__);
      } 
      else if(pieces.size() > 1){
        mainpiece = scinew UnionGeometryPiece(pieces);
      } 
      else {
        mainpiece = pieces[0];
      }

      d_scalar->interiorRegions.push_back(scinew interiorRegion(mainpiece, geom_obj_ps));
    }
  }
}

//______________________________________________________________________
//
void PassiveScalar::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP model_ps = ps->appendChild("Model");
  model_ps->setAttribute( "type","PassiveScalar" );
  ProblemSpecP PS_ps = model_ps->appendChild( "PassiveScalar" );

  PS_ps->appendElement( "material",d_matl->getName() );
  ProblemSpecP scalar_ps = PS_ps->appendChild( "scalar" );
  scalar_ps->setAttribute( "name", "f" );
  scalar_ps->appendElement( "test_conservation", d_test_conservation );


  ProblemSpecP const_ps = scalar_ps->appendChild( "constants" );
  const_ps->appendElement( "initialize_diffusion_knob",d_scalar->initialize_diffusion_knob) ;
  const_ps->appendElement( "diffusivity",              d_scalar->diff_coeff );
  const_ps->appendElement( "AMR_Refinement_Criteria",  d_scalar->refineCriteria );

  //__________________________________
  //  initialization regions
  ProblemSpecP init_ps = scalar_ps->appendChild( "initialization" );
  
  vector<Region*>::const_iterator iter;
  for ( iter = d_scalar->regions.begin(); iter != d_scalar->regions.end(); iter++) {
    ProblemSpecP geom_ps = init_ps->appendChild( "geom_object" );
    
    (*iter)->piece->outputProblemSpec(geom_ps);
    geom_ps->appendElement( "scalar",(*iter)->initialScalar );
  }
  

  //__________________________________
  //  regions inside the domain
  if( d_scalar->interiorRegions.size() > 0 ){
    ProblemSpecP int_ps = scalar_ps->appendChild( "interiorSources" );
    
    vector<interiorRegion*>::const_iterator itr;
    for ( itr = d_scalar->interiorRegions.begin(); itr != d_scalar->interiorRegions.end(); itr++) {
      ProblemSpecP geom_ps = int_ps->appendChild("geom_object");
      (*itr)->piece->outputProblemSpec( geom_ps );
      
      geom_ps->appendElement( "scalar",   (*itr)->value );
      geom_ps->appendElement( "maxScalar",(*itr)->clampValue );
    }
  }
}


//______________________________________________________________________
//      S C H E D U L E   I N I T I A L I Z E
void PassiveScalar::scheduleInitialize(SchedulerP& sched,
                                       const LevelP& level)
{
  printSchedule(level,dout_models_ps,"PassiveScalar::scheduleInitialize");
  
  Task* t = scinew Task("PassiveScalar::initialize", 
                  this, &PassiveScalar::initialize);
  
  t->requires(Task::NewDW, Ilb->timeStepLabel );
  t->computes(d_scalar->scalar_CCLabel);
  
  sched->addTask(t, level->eachPatch(), d_matl_set);
}
//______________________________________________________________________
//       I N I T I A L I Z E
void PassiveScalar::initialize(const ProcessorGroup*, 
                               const PatchSubset* patches,
                               const MaterialSubset*,
                               DataWarehouse*,
                               DataWarehouse* new_dw)
{
  timeStep_vartype timeStep;
  new_dw->get(timeStep, VarLabel::find( timeStep_name) );

  bool isNotInitialTimeStep = (timeStep > 0);
  
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    
    printTask(patches, patch, dout_models_ps, "Doing PassiveScalar::initialize");
    
    int indx = d_matl->getDWIndex();
    
    CCVariable<double>  f;
    new_dw->allocateAndPut(f, d_scalar->scalar_CCLabel, indx, patch);

    f.initialize(0);
    
    //__________________________________
    //  Uniform initialization scalar field in a region
    for(vector<Region*>::iterator iter = d_scalar->regions.begin();
                                  iter != d_scalar->regions.end(); iter++){
      Region* region = *iter;
      
      if(region->uniformInitialize){
      
        for(CellIterator iter = patch->getExtraCellIterator(); !iter.done(); iter++){
          IntVector c = *iter;
          
          Point p = patch->cellPosition(c);            
          if(region->piece->inside(p)) {
            f[c] = region->initialScalar;
          }
        } // Over cells
      }

      //__________________________________
      // Sinusoidal & linear initialization
      if(!region->uniformInitialize){
        IntVector freq = region->freq;
        
        // bulletproofing
        if(region->sinusoidalInitialize && freq.x()==0 && freq.y()==0 && freq.z()==0){
          throw ProblemSetupException("PassiveScalar: you need to specify a <freq> whenever you use sinusoidalInitialize", __FILE__, __LINE__);
        }
        
        Vector slope = region->slope;
        if((region->linearInitialize || region->triangularInitialize) && slope.x()==0 && slope.y()==0 && slope.z()==0){
          throw ProblemSetupException("PassiveScalar: you need to specify a <slope> whenever you use linearInitialize", __FILE__, __LINE__);
        }
        
        Vector coeff = region->coeff;
        cerr<<"coeff"<<region->coeff<<endl;
        if( (region->quadraticInitialize || region->exponentialInitialize_1D ||  region->exponentialInitialize_2D)
           && coeff.x()==0 && coeff.y()==0 && coeff.z()==0){
          cerr<<"coeff"<<coeff<<endl;
          throw ProblemSetupException("PassiveScalar: you need to specify a <coeff> for this initialization", __FILE__, __LINE__);
        }

        if(region->exponentialInitialize_1D &&  ( (coeff.x()*coeff.y()!=0) || (coeff.y()*coeff.z()!=0) || (coeff.x()*coeff.z()!=0) )  ) {
          throw ProblemSetupException("PassiveScalar: 1D Exponential Initialize. This profile is designed for 1D problems only. Try exponentialInitialize_2D instead",__FILE__, __LINE__);
        }
          

        if(region->exponentialInitialize_2D && (coeff.x()!=0) && (coeff.y()!=0) && (coeff.z()!=0) ) {
          throw ProblemSetupException("PassiveScalar: 2D Exponential Initialize. This profile is designed for 2D problems only, one <coeff> must equal zero",__FILE__, __LINE__);
        }
        
        Point lo = region->piece->getBoundingBox().lower();
        Point hi = region->piece->getBoundingBox().upper();
        Vector dist = hi.asVector() - lo.asVector();
        
        //__________________________________
        //
        for(CellIterator iter = patch->getExtraCellIterator(); !iter.done(); iter++){
        
          IntVector c = *iter;
        
          Point p = patch->cellPosition(c);            
          if(region->piece->inside(p)) {
            // normalized distance
            Vector d = (p.asVector() - lo.asVector() )/dist;
            
            if(region->sinusoidalInitialize){
              f[c] = sin( 2.0 * freq.x() * d.x() * M_PI) + 
                     sin( 2.0 * freq.y() * d.y() * M_PI)  + 
                     sin( 2.0 * freq.z() * d.z() * M_PI);
            }
            if(region->linearInitialize){  // f[c] = kx + b
              f[c] = (slope.x() * d.x() + slope.y() * d.y() + slope.z() * d.z() ) +  region->initialScalar; 
            }
            if(region->triangularInitialize){
              if(d.x() <= 0.5)
                f[c] = slope.x()*d.x();
              else
                f[c] = slope.x()*(1.0-d.x());
            }
            if(region->quadraticInitialize){
              if(d.x() <= 0.5)
                f[c] = pow(d.x(),2) - d.x();
              else{
                f[c] = pow( (1.0 - d.x()),2) - d.x();
              } 
            }
            if(region->cubicInitialize){    
              if(d.x() <= 0.5)
                f[c] = -1.3333333*pow(d.x(),3)  + pow(d.x(),2);
              else{
                f[c] = -1.3333333*pow( (1.0 - d.x()),3) + pow( (1.0 - d.x()),2);
              } 
            }
            
            // This is a 2-D profile        
            if(region->exponentialInitialize_2D) {
              double coeff1 = 0., coeff2 = 0. ,  d1 = 0. , d2= 0.;
              if (coeff.x()==0) {
                coeff1 = coeff.y();
                coeff2 = coeff.z();
                d1 = d.y();
                d2 = d.z();
              }
              else if (coeff.y()==0) {
                coeff1 = coeff.x();
                coeff2 = coeff.z();
                d1 = d.x();
                d2 = d.z();
              }
              else if (coeff.z()==0) {
                coeff1 = coeff.y();
                coeff2 = coeff.x();
                d1 = d.y();
                d2 = d.x();
              }
              f[c] = coeff1 * exp(-1.0/( d1 * ( 1.0 - d1 ) + 1e-100) )
                   * coeff2 * exp(-1.0/( d2 * ( 1.0 - d2 ) + 1e-100) );
            }

            // This is a 1-D profile - Donot use it for 2-D
            
            if (region->exponentialInitialize_1D ){
              f[c] = coeff.x() * exp(-1.0/( d.x() * ( 1.0 - d.x() ) + 1e-100) )
                   + coeff.y() * exp(-1.0/( d.y() * ( 1.0 - d.y() ) + 1e-100) )
                   + coeff.z() * exp(-1.0/( d.z() * ( 1.0 - d.z() ) + 1e-100) );
            }
          }
        }
      }  // sinusoidal Initialize  
    } // regions
    setBC(f,"scalar-f", patch, m_materialManager,indx, new_dw, isNotInitialTimeStep); 
  }  // patches
}

//______________________________________________________________________     
void PassiveScalar::scheduleModifyThermoTransportProperties(SchedulerP& sched,
                                                            const LevelP& level,
                                                            const MaterialSet* /*ice_matls*/)
{
  printSchedule(level,dout_models_ps,"PassiveScalar::scheduleModifyThermoTransportProperties");
  
  Task* t = scinew Task("PassiveScalar::modifyThermoTransportProperties", 
                   this,&PassiveScalar::modifyThermoTransportProperties);
                   
  t->computes(d_scalar->diffusionCoefLabel);
  sched->addTask(t, level->eachPatch(), d_matl_set);
}
//______________________________________________________________________
void PassiveScalar::modifyThermoTransportProperties(const ProcessorGroup*, 
                                                    const PatchSubset* patches,
                                                    const MaterialSubset*,
                                                    DataWarehouse* /*old_dw*/,
                                                    DataWarehouse* new_dw)
{ 
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    
    printTask(patches, patch, dout_models_ps, "Doing PassiveScalar::modifyThermoTransportProperties");
   
    int indx = d_matl->getDWIndex();
    CCVariable<double> diffusionCoeff;
    new_dw->allocateAndPut(diffusionCoeff, 
                           d_scalar->diffusionCoefLabel,indx, patch);  
        
    diffusionCoeff.initialize(d_scalar->diff_coeff);
  }
} 

//______________________________________________________________________
void PassiveScalar::computeSpecificHeat(CCVariable<double>& ,
                                        const Patch* ,
                                        DataWarehouse* ,
                                        const int )
{ 
  //none
} 

//______________________________________________________________________
void PassiveScalar::scheduleComputeModelSources(SchedulerP& sched,
                                                const LevelP& level)
{
  printSchedule(level,dout_models_ps,"PassiveScalar::scheduleComputeModelSources");
  
  Task* t = scinew Task("PassiveScalar::computeModelSources", 
                   this,&PassiveScalar::computeModelSources);
                     
  Ghost::GhostType  gac = Ghost::AroundCells;
  t->requires(Task::OldDW, Ilb->delTLabel, level.get_rep() );
  t->requires(Task::NewDW, d_scalar->diffusionCoefLabel, gac,1);
  t->requires(Task::OldDW, d_scalar->scalar_CCLabel,     gac,1); 
  t->modifies(d_scalar->scalar_source_CCLabel);

  sched->addTask(t, level->eachPatch(), d_matl_set);
}

//______________________________________________________________________
void PassiveScalar::computeModelSources(const ProcessorGroup*, 
                                        const PatchSubset* patches,
                                        const MaterialSubset* /*matls*/,
                                        DataWarehouse* old_dw,
                                        DataWarehouse* new_dw)
{
  const Level* level = getLevel(patches);
  delt_vartype delT;
  old_dw->get(delT, Ilb->delTLabel, level);     
  Ghost::GhostType gac = Ghost::AroundCells;
  
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    
    printTask(patches, patch, dout_models_ps, "Doing PassiveScalar::computeModelSources");
   
    constCCVariable<double> f_old, diff_coeff;
    CCVariable<double>  f_src;
    
    int indx = d_matl->getDWIndex();
    old_dw->get(f_old,      d_scalar->scalar_CCLabel,      indx, patch, gac,1);
    new_dw->get(diff_coeff, d_scalar->diffusionCoefLabel,  indx, patch, gac,1);
    new_dw->allocateAndPut(f_src, d_scalar->scalar_source_CCLabel,
                                                           indx, patch);
    f_src.initialize(0.0);
                                                            
    //__________________________________
    //  scalar diffusion
    double diff_coeff_test = d_scalar->diff_coeff;
    if(diff_coeff_test != 0.0){ 
      bool use_vol_frac = false; // don't include vol_frac in diffusion calc.
      CCVariable<double> placeHolder;

      scalarDiffusionOperator(new_dw, patch, use_vol_frac, f_old,
                              placeHolder, f_src, diff_coeff, delT);
    }  
    
    //__________________________________
    //  interior source regions
    for(vector<interiorRegion*>::iterator iter = d_scalar->interiorRegions.begin();
                                          iter != d_scalar->interiorRegions.end(); iter++){
      interiorRegion* region = *iter;
            
      for(CellIterator iter = patch->getExtraCellIterator(); !iter.done(); iter++){
        IntVector c = *iter;
        
        Point p = patch->cellPosition(c);            
        
        if(region->piece->inside(p)) {
          f_src[c] = region->value;
          
          double f_test = f_old[c] + f_src[c];
          double clamp = region->clampValue;
          
          if (f_test > clamp ){
            f_src[c] = clamp - f_old[c];
          }
        }
      } // Over cells
    }  //interiorRegions
  }  // patches
}
//__________________________________      
void PassiveScalar::scheduleComputeStableTimeStep(SchedulerP&,
                                                  const LevelP&)
{
  // None necessary...
}

//______________________________________________________________________
void PassiveScalar::scheduleTestConservation(SchedulerP& sched,
                                             const PatchSet* patches)
{
  const Level* level = getLevel(patches);
  int L = level->getIndex();
  
  if(d_test_conservation && L == 0){

    printSchedule(patches, dout_models_ps, "PassiveScalar::scheduleTestConservation");
    
    Task* t = scinew Task("PassiveScalar::testConservation", 
                     this,&PassiveScalar::testConservation);

    Ghost::GhostType  gn = Ghost::None;
    // compute sum(scalar_f * mass)
    t->requires(Task::OldDW, Ilb->delTLabel, getLevel(patches) );
    t->requires(Task::NewDW, d_scalar->scalar_CCLabel, gn,0); 
    t->requires(Task::NewDW, Ilb->rho_CCLabel,          gn,0);
    t->requires(Task::NewDW, Ilb->uvel_FCMELabel,       gn,0); 
    t->requires(Task::NewDW, Ilb->vvel_FCMELabel,       gn,0); 
    t->requires(Task::NewDW, Ilb->wvel_FCMELabel,       gn,0); 
    t->computes(Slb->sum_scalar_fLabel);

    sched->addTask(t, patches, d_matl_set);
  }
}

//______________________________________________________________________
void PassiveScalar::testConservation(const ProcessorGroup*, 
                                     const PatchSubset* patches,
                                     const MaterialSubset* /*matls*/,
                                     DataWarehouse* old_dw,
                                     DataWarehouse* new_dw)
{
  const Level* level = getLevel(patches);
  delt_vartype delT;
  old_dw->get(delT, Ilb->delTLabel, level);     
  Ghost::GhostType gn = Ghost::None; 
  
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    
    printTask(patches, patch, dout_models_ps, "Doing PassiveScalar::testConservation");
                   
    //__________________________________
    //  conservation of f test
    constCCVariable<double> rho_CC, f;
    constSFCXVariable<double> uvel_FC;
    constSFCYVariable<double> vvel_FC;
    constSFCZVariable<double> wvel_FC;
    int indx = d_matl->getDWIndex();
    
    new_dw->get(f,       d_scalar->scalar_CCLabel,indx,patch,gn,0);
    new_dw->get(rho_CC,  Ilb->rho_CCLabel,         indx,patch,gn,0); 
    new_dw->get(uvel_FC, Ilb->uvel_FCMELabel,      indx,patch,gn,0); 
    new_dw->get(vvel_FC, Ilb->vvel_FCMELabel,      indx,patch,gn,0); 
    new_dw->get(wvel_FC, Ilb->wvel_FCMELabel,      indx,patch,gn,0); 
    
    Vector dx = patch->dCell();
    double cellVol = dx.x()*dx.y()*dx.z();

    CCVariable<double> q_CC;
    new_dw->allocateTemporary(q_CC, patch);

    for(CellIterator iter = patch->getExtraCellIterator(); !iter.done(); iter++) {
      IntVector c = *iter;
      q_CC[c] = rho_CC[c]*cellVol*f[c];
    }

    double sum_mass_f;
    conservationTest<double>(patch, delT, q_CC, uvel_FC, vvel_FC, wvel_FC, sum_mass_f);

    new_dw->put(sum_vartype(sum_mass_f), Slb->sum_scalar_fLabel);
  }
}

//______________________________________________________________________
//
void PassiveScalar::scheduleErrorEstimate(const LevelP& coarseLevel,
                                          SchedulerP& sched)
{
  printSchedule( coarseLevel,dout_models_ps,"PassiveScalar::scheduleErrorEstimate" );
  
  Task* t = scinew Task("PassiveScalar::errorEstimate", 
                  this, &PassiveScalar::errorEstimate, false);  
  
  Ghost::GhostType  gac  = Ghost::AroundCells; 
  
  t->requires(Task::NewDW, d_scalar->scalar_CCLabel,  d_matl_sub, gac,1);
  
  t->computes(d_scalar->mag_grad_scalarLabel, d_matl_sub);
  t->modifies(m_regridder->getRefineFlagLabel(),      m_regridder->refineFlagMaterials());
  t->modifies(m_regridder->getRefinePatchFlagLabel(), m_regridder->refineFlagMaterials());
 
  // define the material set of 0 and whatever the passive scalar index is 
  // don't add matl 0 twice 
  MaterialSet* matl_set;
  vector<int> m;
  m.push_back(0);
  if(d_matl->getDWIndex() != 0){
    m.push_back(d_matl->getDWIndex());
  }
  matl_set = scinew MaterialSet();
  matl_set->addAll(m);
  matl_set->addReference();
    
  sched->addTask(t, coarseLevel->eachPatch(), matl_set); 
}
/*_____________________________________________________________________
 Function~  PassiveScalar::errorEstimate--
______________________________________________________________________*/
void PassiveScalar::errorEstimate(const ProcessorGroup*,
                                  const PatchSubset* patches,
                                  const MaterialSubset*,
                                  DataWarehouse*,
                                  DataWarehouse* new_dw,
                                  bool)
{  
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    printTask(patches, patch, dout_models_ps, "Doing PassiveScalar::errorEstimate");
    
    Ghost::GhostType  gac  = Ghost::AroundCells;
    const VarLabel* refineFlagLabel = m_regridder->getRefineFlagLabel();
    const VarLabel* refinePatchLabel= m_regridder->getRefinePatchFlagLabel();
    
    CCVariable<int> refineFlag;
    new_dw->getModifiable(refineFlag, refineFlagLabel, 0, patch);      

    PerPatch<PatchFlagP> refinePatchFlag;
    new_dw->get(refinePatchFlag, refinePatchLabel, 0, patch);

    int indx = d_matl->getDWIndex();
    constCCVariable<double> f;
    CCVariable<double> mag_grad_f;
    
    new_dw->get(f,                     d_scalar->scalar_CCLabel,       indx ,patch,gac,1);
    new_dw->allocateAndPut(mag_grad_f, d_scalar->mag_grad_scalarLabel, indx, patch);
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
