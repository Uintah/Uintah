/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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

#include <CCA/Components/ICE/ICEMaterial.h>
#include <CCA/Components/ICE/ConservationTest.h>
#include <CCA/Components/ICE/Diffusion.h>
#include <CCA/Components/Models/FluidsBased/flameSheet_rxn.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Exceptions/ParameterNotFound.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Grid/Box.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Material.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/GeometryPiece/GeometryPieceFactory.h>
#include <Core/GeometryPiece/UnionGeometryPiece.h>
#include <Core/Labels/ICELabel.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/ProblemSpec/ProblemSpec.h>

#include <Core/Containers/StaticArray.h>
#include <Core/Math/MiscMath.h>
#include <iostream>
#include <Core/Util/DebugStream.h>
#include <cstdio>

using namespace Uintah;
using namespace std;
static DebugStream cout_doing("MODELS_DOING_COUT", false);

//______________________________________________________________________
// flame sheet approach for laminar diffusion flames.
// Reference:  "An Introduction to Combustion Concepts and Applications"
//              by Stephen Turns pp 268 - 275
// Assumptions: cp_fuel = cp_oxidizer = cp_products
//              Enthalpies of formation of the oxidizer and products  0.0
//              Thus the enthalpy of formatiom of the fuel equal the heat
//              of combustion.
//              Thermal energy and species diffusivities are equal
//               

flameSheet_rxn::flameSheet_rxn(const ProcessorGroup* myworld, 
                               ProblemSpecP& params)
  : ModelInterface(myworld), params(params)
{
  d_matl_set = 0;
  lb  = scinew ICELabel();
}
//__________________________________
flameSheet_rxn::~flameSheet_rxn()
{
  if(d_matl_set && d_matl_set->removeReference()){
    delete d_matl_set;
  }

  VarLabel::destroy(d_scalar->scalar_CCLabel);
  VarLabel::destroy(d_scalar->scalar_source_CCLabel);
  VarLabel::destroy(d_scalar->sum_scalar_fLabel);
  delete lb;
    
  for(vector<Region*>::iterator iter = d_scalar->regions.begin();
     iter != d_scalar->regions.end(); iter++){
    Region* region = *iter;
    delete region;
  }
}
//__________________________________
//
flameSheet_rxn::Region::Region(GeometryPieceP piece, ProblemSpecP& ps)
  : piece(piece)
{
  ps->require("scalar", initialScalar);
}
//______________________________________________________________________
//    Problem Setup
void flameSheet_rxn::problemSetup(GridP&, SimulationStateP& in_state,
                           ModelSetup* setup)
{
  cout_doing << "Doing problemSetup \t\t\t\tFLAMESHEET" << endl;
  d_sharedState = in_state;
  d_matl = d_sharedState->parseAndLookupMaterial(params, "material");

  vector<int> m(1);
  m[0] = d_matl->getDWIndex();
  d_matl_set = scinew MaterialSet();
  d_matl_set->addAll(m);
  d_matl_set->addReference();

  // determine the specific heat of that matl.
  Material* matl = d_sharedState->getMaterial( m[0] );
  ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
  if (ice_matl){
    d_cp = ice_matl->getSpecificHeat();
  }   
  
  d_scalar = scinew Scalar();
  d_scalar->index = 0;
  d_scalar->name  = "f";
  
  const TypeDescription* td_CCdouble = CCVariable<double>::getTypeDescription();
  d_scalar->scalar_CCLabel =        VarLabel::create("scalar-f",    td_CCdouble);
  d_scalar->scalar_source_CCLabel = VarLabel::create("scalar_f_src",td_CCdouble);
  d_scalar->sum_scalar_fLabel     =  VarLabel::create("sum_scalar_f", 
                                            sum_vartype::getTypeDescription());
                                            
  setup->registerTransportedVariable(d_matl_set,
                                    d_scalar->scalar_CCLabel,
                                    d_scalar->scalar_source_CCLabel);

 

  //__________________________________
  // Read in the constants for the scalar 
  ProblemSpecP child = params->findBlock("scalar");
  if (!child){
    throw ProblemSetupException("flameSheet: Couldn't find scalar tag", __FILE__, __LINE__);    
  }
  
  child->getWithDefault("test_conservation", d_test_conservation, false);
 
  //  reaction constants
  ProblemSpecP react_ps = child->findBlock("reaction_constants");
  if(!react_ps) {
    throw ProblemSetupException("Cannot find reaction_constants tag", __FILE__, __LINE__);
  }

  react_ps->getWithDefault( "f_stoichometric",       d_f_stoic,       -9.0 );
  react_ps->getWithDefault( "delta_H_combustion",    d_del_h_comb,    -9.0 );
  react_ps->getWithDefault( "oxidizer_temp_infinity",d_T_oxidizer_inf,-9.0 );
  react_ps->getWithDefault( "initial_fuel_temp",     d_T_fuel_init,   -9.0 );
  react_ps->getWithDefault( "diffusivity",           d_diffusivity,   -9.0 );
  react_ps->getWithDefault( "smear_initialDistribution_knob",       
                                                     d_smear_initialDistribution_knob,  0 );
                            
  if( Floor(d_f_stoic) == -9        ||  Floor(d_del_h_comb) == -9 ||    // bulletproofing
      Floor(d_T_oxidizer_inf) == -9 ||  Floor(d_T_fuel_init) == -9 ) {
    ostringstream warn;
    warn << " ERROR FlameSheet_rxn: Input variable(s) not specified \n" 
         << "\n f_stoichometric        "<< d_f_stoic
         << "\n delta_H_combustion     "<< d_del_h_comb
         << "\n oxidizer_temp_infinity "<< d_T_oxidizer_inf
         << "\n fuel_temp_init         "<< d_T_fuel_init 
         << "\n diffusivity            "<< d_diffusivity<< endl;
    throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
  }

  //__________________________________
  //  geom objects
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
}

void flameSheet_rxn::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP model_ps = ps->appendChild("Model");
  model_ps->setAttribute("type","flameSheet_rxn");

  model_ps->appendElement("material",d_matl->getName());
  ProblemSpecP scalar_ps = model_ps->appendChild("scalar");
  scalar_ps->setAttribute("name","f");
  scalar_ps->appendElement("test_conservation",d_test_conservation);

  ProblemSpecP const_ps = scalar_ps->appendChild("reaction_constants");
  const_ps->appendElement("f_stoichometric",       d_f_stoic);  
  const_ps->appendElement("delta_H_combustion",    d_del_h_comb);  
  const_ps->appendElement("oxidizer_temp_infinity",d_T_oxidizer_inf);
  const_ps->appendElement("initial_fuel_temp",     d_T_fuel_init);    
  const_ps->appendElement("diffusivity",           d_diffusivity);

  const_ps->appendElement("smear_initialDistribution_knob",
                            d_smear_initialDistribution_knob);   

  
  for (vector<Region*>::const_iterator it = d_scalar->regions.begin();
       it != d_scalar->regions.end(); it++) {
    ProblemSpecP geom_ps = scalar_ps->appendChild("geom_object");
    (*it)->piece->outputProblemSpec(geom_ps);
    geom_ps->appendElement("scalar",(*it)->initialScalar);
  }

}

//______________________________________________________________________
void flameSheet_rxn::scheduleInitialize(SchedulerP& sched,
                                            const LevelP& level,
                                            const ModelInfo*)
{
  //__________________________________
  //  intialize the scalar field
  cout_doing << "FLAMESHEET::scheduleInitialize " << endl;
  Task* t = scinew Task("flameSheet_rxn::initialize",
                  this, &flameSheet_rxn::initialize);
 t->computes(d_scalar->scalar_CCLabel);
  sched->addTask(t, level->eachPatch(), d_matl_set);
}
//______________________________________________________________________
void flameSheet_rxn::initialize(const ProcessorGroup*, 
                                   const PatchSubset* patches,
                                   const MaterialSubset* matls,
                                   DataWarehouse*,
                                   DataWarehouse* new_dw)
{
  cout_doing << "Doing Initialize \t\t\t\tFLAMESHEET" << endl;
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    int indx = d_matl->getDWIndex();

    CCVariable<double>  f;
    new_dw->allocateAndPut(f, d_scalar->scalar_CCLabel, indx, patch);
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
    CCVariable<double> FakeDiffusivity;
    new_dw->allocateTemporary(FakeDiffusivity, patch);
    FakeDiffusivity.initialize(1.0);    //  HARDWIRED
    double fakedelT = 1.0;

    for( int i =1 ; i < d_smear_initialDistribution_knob; i++ ){
      bool use_vol_frac = false; // don't include vol_frac in diffusion calc.
      CCVariable<double> placeHolder;
      scalarDiffusionOperator(new_dw, patch, use_vol_frac,  f,
                              placeHolder,f, FakeDiffusivity, fakedelT); 
    }  // diffusion loop
  }  // patches
}


//__________________________________
void flameSheet_rxn::scheduleComputeModelSources(SchedulerP& sched,
                                                      const LevelP& level,
                                                      const ModelInfo* mi)
{
  cout_doing << "FLAMESHEET::scheduleComputeModelSources " << endl;
  Task* t = scinew Task("flameSheet_rxn::computeModelSources",this, 
                        &flameSheet_rxn::computeModelSources, mi);
                     
  Ghost::GhostType  gn = Ghost::None;  
  Ghost::GhostType  gac = Ghost::AroundCells; 
  t->modifies(mi->modelEng_srcLabel);
  t->requires(Task::OldDW, mi->rho_CCLabel,         gn);
  t->requires(Task::OldDW, mi->temp_CCLabel,        gn);
  t->requires(Task::NewDW, mi->specific_heatLabel,  gn);
  t->requires(Task::NewDW, mi->gammaLabel,          gn);
  t->requires(Task::OldDW, d_scalar->scalar_CCLabel, gac,1);
  //t->requires(Task::OldDW, mi->delT_Label);   AMR
  
  t->modifies(d_scalar->scalar_source_CCLabel);
  sched->addTask(t, level->eachPatch(), d_matl_set);
}
//______________________________________________________________________
void flameSheet_rxn::computeModelSources(const ProcessorGroup*, 
                                           const PatchSubset* patches,
                                           const MaterialSubset* matls,
                                           DataWarehouse* old_dw,
                                           DataWarehouse* new_dw,
                                           const ModelInfo* mi)
{
  const Level* level = getLevel(patches);
  delt_vartype delT;
  old_dw->get(delT, mi->delT_Label,level);
 
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    cout_doing << "Doing computeModelSources on patch "<<patch->getID()
               << "\t\t\t\t\t FLAMESHEET" << endl;


    Ghost::GhostType  gn = Ghost::None;  
    Ghost::GhostType  gac = Ghost::AroundCells;     

    int indx = d_matl->getDWIndex();
    constCCVariable<double> rho_CC;
    constCCVariable<double> Temp_CC;
    constCCVariable<double> f_old; 
    constCCVariable<double> cv; 
    constCCVariable<double> gamma;
    CCVariable<double> energySource;
    CCVariable<double> f_src;
    
    old_dw->get(rho_CC,      mi->rho_CCLabel,         indx, patch, gn, 0);
    old_dw->get(Temp_CC,     mi->temp_CCLabel,        indx, patch, gn, 0);
    new_dw->get(cv,          mi->specific_heatLabel,  indx, patch, gn, 0);
    new_dw->get(gamma,       mi->gammaLabel,          indx, patch, gn, 0);
    new_dw->getModifiable(energySource,   
                             mi->modelEng_srcLabel,indx,patch);
                     
    old_dw->get(f_old,            d_scalar->scalar_CCLabel,       
                                                      indx, patch, gac, 1);
    new_dw->allocateAndPut(f_src, d_scalar->scalar_source_CCLabel,
                                                      indx, patch, gn, 0);

    //__________________________________
    //   G R O S S N E S S
    double nu             = (1.0/d_f_stoic) - 1.0;
  //double d_del_h_comb   = 1000.0* 74831.0;    // Enthalpy of combustion J/kg
/*`==========TESTING==========*/        // for variable cp this should be changed
    double del_h_comb = d_del_h_comb * d_cp/d_f_stoic;   
/*==========TESTING==========`*/   
    double fuzzyOne = 1.0 + 1e-10;
    double fuzzyZero = 0.0 - 1e10;
    int     numCells = 0, sum = 0;
    Vector dx = patch->dCell();
    double volume = dx.x()*dx.y()*dx.z();
    
    //__________________________________   
    for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
      IntVector c = *iter;
      
      double newTemp = -999;
      double Y_fuel  = -999;
      double Y_products = -999;
      double cp      = gamma[c] * cv[c];
      double mass    = rho_CC[c]*volume;
      double f       = f_old[c];
      double oldTemp = Temp_CC[c];
      numCells++;

      f = min(f,1.0);    // keep 0 < f < 1
      f = max(f,0.0);
      //__________________________________
      // compute the energy source
      //__________________________________
      //  Inside the flame    
      if (d_f_stoic < f && f <= fuzzyOne ){  
        sum++;         
        Y_fuel     = (f - d_f_stoic)/(1.0 - d_f_stoic); 
        Y_products = (1 - f)/(1-d_f_stoic);         // eqs 9.43a,b,c & 9.51a

        double tmp = d_f_stoic * del_h_comb/((1.0 - d_f_stoic) * cp);
        double A   = f * ( (d_T_fuel_init - d_T_oxidizer_inf) - tmp ); 
        newTemp    =  A + d_T_oxidizer_inf + tmp;

      }
      //__________________________________
      //  At the flame surface        
      if (d_f_stoic == f ){ 
        sum++;                         
        Y_fuel     = 0.0;                            // eqs 9.45a,b,c & 9.51a
        Y_products = 1.0;

        double A = d_f_stoic *( del_h_comb/cp + d_T_fuel_init 
                                - d_T_oxidizer_inf);
        newTemp = A + d_T_oxidizer_inf;
      }
      //__________________________________
      //  outside the flame
      if (fuzzyZero <= f && f < d_f_stoic ){ 
        sum++;     
        Y_fuel     = 0.0;                            // eqs 9.46a,b,c & 9.51c
        Y_products = f/d_f_stoic;

        double A = f *( (del_h_comb/cp) + d_T_fuel_init - d_T_oxidizer_inf);
        newTemp  = A + d_T_oxidizer_inf;
      }  

      double new_f =Y_fuel + Y_products/(1.0 + nu);        // eqs 7.54         
      double energyx =( newTemp - oldTemp) * cp * mass;
      energySource[c] += energyx;
      f_src[c] += new_f - f;
    }  //iter

    //__________________________________
    //  bulletproofing
    if (sum != numCells) {
      ostringstream warn;
      warn << "ERROR: flameSheet_rxn Model: invalid value for f "
           << "somewhere in the scalar field: "<< sum
           << " cells were touched out of "<< numCells
           << " Total cells ";
      throw InvalidValue(warn.str(), __FILE__, __LINE__);
    }         
    //__________________________________
    //  Tack on diffusion
    if(d_diffusivity != 0.0){ 
      bool use_vol_frac = false; // don't include vol_frac in diffusion calc.
      CCVariable<double> placeHolder;
      /*`==========TESTING==========*/    
      // this needs to be changed
      CCVariable<double> diff_coeff;
      new_dw->allocateTemporary(diff_coeff, patch,Ghost::AroundCells, 1);
      diff_coeff.initialize(d_diffusivity);    
      /*==========TESTING==========`*/
      scalarDiffusionOperator(new_dw, patch, use_vol_frac,f_old,
                              placeHolder,f_src, diff_coeff, delT);
    }  // diffusivity > 0 

  }
}
//______________________________________________________________________
void flameSheet_rxn::scheduleTestConservation(SchedulerP& sched,
                                            const PatchSet* patches,
                                            const ModelInfo* mi)
{
  if(d_test_conservation){
    cout_doing << "PASSIVESCALAR::scheduleTestConservation " << endl;
    Task* t = scinew Task("flameSheet_rxn::testConservation", 
                     this,&flameSheet_rxn::testConservation, mi);

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
void flameSheet_rxn::testConservation(const ProcessorGroup*, 
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
               << "\t\t\t FLAMESHEET" << endl;
               
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
void flameSheet_rxn::scheduleComputeStableTimestep(SchedulerP&,
                                           const LevelP&,
                                           const ModelInfo*)
{
  // None necessary...
}
//______________________________________________________________________
void flameSheet_rxn::scheduleModifyThermoTransportProperties(SchedulerP&,
                                                        const LevelP&,
                                                        const MaterialSet*)
{
  // do nothing      
}
void flameSheet_rxn::computeSpecificHeat(CCVariable<double>&,
                                 const Patch*,
                                 DataWarehouse*,
                                 const int)
{
  //do nothing
}
//______________________________________________________________________
//
void flameSheet_rxn::scheduleErrorEstimate(const LevelP&,
                                           SchedulerP&)
{
  // Not implemented yet
}
