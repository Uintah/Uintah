#include <Packages/Uintah/CCA/Components/ICE/ICEMaterial.h>
#include <Packages/Uintah/CCA/Components/ICE/Diffusion.h>
#include <Packages/Uintah/CCA/Components/Models/test/SimpleRxn.h>
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
#include <Packages/Uintah/Core/Exceptions/ParameterNotFound.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/Core/Exceptions/InvalidValue.h>
#include <Core/Containers/StaticArray.h>
#include <Core/Math/MiscMath.h>
#include <iostream>
#include <Core/Util/DebugStream.h>
#include <stdio.h>

using namespace Uintah;
using namespace std;
//__________________________________
//  To turn on the output
//  setenv SCI_DEBUG "SIMPLE_RXN_DOING_COUT:+,SIMPLE_RXN_DBG_COUT:+"
//  SIMPLE_RXN_DBG:  dumps out during problemSetup 
//  SIMPLE_RXN_DOING_COUT:   dumps when tasks are scheduled and performed
static DebugStream cout_doing("SIMPLE_RXN_DOING_COUT", false);
static DebugStream cout_dbg("SIMPLE_RXN_DBG_COUT", false);
//______________________________________________________________________              
SimpleRxn::SimpleRxn(const ProcessorGroup* myworld, 
                               ProblemSpecP& params)
  : ModelInterface(myworld), params(params)
{
  d_matl_set = 0;
  lb = scinew ICELabel();
}

//__________________________________
SimpleRxn::~SimpleRxn()
{
  if(d_matl_set && d_matl_set->removeReference()) {
    delete d_matl_set;
  }
  delete lb;
    
  VarLabel::destroy(d_scalar->scalar_CCLabel);
  VarLabel::destroy(d_scalar->scalar_source_CCLabel);
  VarLabel::destroy(d_scalar->diffusionCoefLabel);
  
  for(vector<Region*>::iterator iter = d_scalar->regions.begin();
                                iter != d_scalar->regions.end(); iter++){
    Region* region = *iter;
    delete region->piece;
    delete region;

  }
}

//__________________________________
SimpleRxn::Region::Region(GeometryPiece* piece, ProblemSpecP& ps)
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
  d_matl_set = new MaterialSet();
  d_matl_set->addAll(m);
  d_matl_set->addReference();

  // determine the specific heat of that matl.
  Material* matl = sharedState->getMaterial( m[0] );
  ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
  if (ice_matl){
    d_cp = ice_matl->getSpecificHeat();
  }   
  //__________________________________
  // - create Label names
  // - register the scalar to be transported
  d_scalar = new Scalar();
  d_scalar->index = 0;
  d_scalar->name  = "f";
  
  const TypeDescription* td_CCdouble = CCVariable<double>::getTypeDescription();
  d_scalar->scalar_CCLabel =     VarLabel::create("scalar-f",       td_CCdouble);
  d_scalar->diffusionCoefLabel = VarLabel::create("scalar-diffCoef",td_CCdouble);
  d_scalar->scalar_source_CCLabel = 
                                 VarLabel::create("scalar-f_src",  td_CCdouble);


  setup->registerTransportedVariable(d_matl_set->getSubset(0),
                                     d_scalar->scalar_CCLabel,
                                     d_scalar->scalar_source_CCLabel);  
  //__________________________________
  // Read in the constants for the scalar
   ProblemSpecP child = params->findBlock("scalar");
   if (!child){
     throw ProblemSetupException("SimpleRxn: Couldn't find scalar tag");    
   }
   ProblemSpecP const_ps = child->findBlock("constants");
   if(!const_ps) {
     throw ProblemSetupException("SimpleRxn: Couldn't find constants tag");
   }
    
  const_ps->getWithDefault("f_stoichometric",d_scalar->f_stoic,    -9);         
  const_ps->getWithDefault("diffusivity",    d_scalar->diff_coeff, -9);       
  const_ps->getWithDefault("initialize_diffusion_knob",       
                            d_scalar->initialize_diffusion_knob,   0);
  const_ps->getWithDefault("rho_air",  d_rho_air,  -9);
  const_ps->getWithDefault("rho_fuel", d_rho_fuel, -9);   
  
  if( d_scalar->f_stoic == -9 || d_rho_air == -9 ||
      d_rho_fuel == -9  ) {
    ostringstream warn;
    warn << " ERROR SimpleRxn: Input variable(s) not specified \n" 
         << "\n f_stoichometric  "<< d_scalar->f_stoic 
         << "\n diffusivity      "<< d_scalar->diff_coeff
         << "\n rho_air          "<< d_rho_air
         << "\n rho_fuel         "<< d_rho_fuel<<endl;
    throw ProblemSetupException(warn.str());
  } 

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
}
//______________________________________________________________________
//      S C H E D U L E   I N I T I A L I Z E
void SimpleRxn::scheduleInitialize(SchedulerP& sched,
                                   const LevelP& level,
                                   const ModelInfo*)
{
  cout_doing << "SIMPLERXN::scheduleInitialize " << endl;
  Task* t = scinew Task("SimpleRxn::initialize", this, &SimpleRxn::initialize);
  t->computes(d_scalar->scalar_CCLabel);
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
    CCVariable<double>  f;
    new_dw->allocateAndPut(f, d_scalar->scalar_CCLabel, indx, patch);
    
    ///__________________________________
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
      constCCVariable<double> placeHolder;
      scalarDiffusionOperator(new_dw, patch, use_vol_frac,
                              placeHolder, placeHolder,  f,
                              f, FakeDiffusivity, fakedelT);
    }
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
  t->computes(d_scalar->diffusionCoefLabel);
  sched->addTask(t, level->eachPatch(), d_matl_set);
}
//______________________________________________________________________
void SimpleRxn::modifyThermoTransportProperties(const ProcessorGroup*, 
                                                const PatchSubset* patches,
                                                const MaterialSubset*,
                                                DataWarehouse*,
                                                DataWarehouse* new_dw)
{ 
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    cout_doing << "Doing modifyThermoTransportProperties on patch "<<patch->getID()<< "\t SIMPLERXN" << endl;
    int indx = d_matl->getDWIndex();
    CCVariable<double> diffusionCoeff;
    new_dw->allocateAndPut(diffusionCoeff, 
                           d_scalar->diffusionCoefLabel,indx, patch);  
    diffusionCoeff.initialize(d_scalar->diff_coeff);
  }
} 

//______________________________________________________________________
void SimpleRxn::scheduleMassExchange(SchedulerP& sched,
                              const LevelP& level,
                              const ModelInfo* mi)
{
  cout_doing << "SIMPLE_RXN::scheduleMassExchange" << endl;
  Task* t = scinew Task("SimpleRxn::massExchange", 
                   this,&SimpleRxn::massExchange, mi);

  t->requires(Task::OldDW, mi->density_CCLabel,  Ghost::None);
  t->modifies(mi->mass_source_CCLabel);
  t->modifies(mi->sp_vol_source_CCLabel);
  sched->addTask(t, level->eachPatch(), d_matl_set);
}
//______________________________________________________________________
void SimpleRxn::massExchange(const ProcessorGroup*, 
                             const PatchSubset* patches,
                             const MaterialSubset*,
                             DataWarehouse* old_dw,
                             DataWarehouse* new_dw,
                             const ModelInfo* mi)
{
  delt_vartype delT;
  old_dw->get(delT, mi->delT_Label);
  
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
void SimpleRxn::scheduleMomentumAndEnergyExchange(SchedulerP& sched,
                                          const LevelP& level,
                                          const ModelInfo* mi)
{
  cout_doing << "SIMPLE_RXN::scheduleMomentumAndEnergyExchange " << endl;
  Task* t = scinew Task("SimpleRxn::momentumAndEnergyExchange", 
                   this,&SimpleRxn::momentumAndEnergyExchange, mi);
                     
  Ghost::GhostType  gn = Ghost::None;  
  Ghost::GhostType  gac = Ghost::AroundCells;
  t->requires(Task::NewDW, d_scalar->diffusionCoefLabel, gac,1);
  t->requires(Task::OldDW, d_scalar->scalar_CCLabel,     gac,1); 
  t->requires(Task::OldDW, mi->density_CCLabel,          gn);
  t->requires(Task::OldDW, mi->temperature_CCLabel,      gn);
  t->requires(Task::OldDW, mi->delT_Label);
  
  t->modifies(mi->momentum_source_CCLabel);
  t->modifies(mi->energy_source_CCLabel);
  t->modifies(d_scalar->scalar_source_CCLabel);
  sched->addTask(t, level->eachPatch(), d_matl_set);
}
//______________________________________________________________________
void SimpleRxn::momentumAndEnergyExchange(const ProcessorGroup*, 
                                            const PatchSubset* patches,
                                            const MaterialSubset* matls,
                                            DataWarehouse* old_dw,
                                            DataWarehouse* new_dw,
                                            const ModelInfo* mi)
{
  delt_vartype delT;
  old_dw->get(delT, mi->delT_Label);
  Ghost::GhostType gn = Ghost::None;         
  Ghost::GhostType gac = Ghost::AroundCells; 
  
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    cout_doing << "Doing momentumAndEnergyExh... on patch "<<patch->getID()<< "\t\tSIMPLERXN" << endl;

    Vector dx = patch->dCell();
    double volume = dx.x()*dx.y()*dx.z();     
    double new_f;
    constCCVariable<double> rho_CC_old,f_old, diff_coeff;
    CCVariable<double> eng_src, sp_vol_src, f_src;
    CCVariable<Vector> mom_src;
    
    int indx = d_matl->getDWIndex();
    old_dw->get(rho_CC_old, mi->density_CCLabel,           indx, patch, gn, 0);
    old_dw->get(f_old,      d_scalar->scalar_CCLabel,      indx, patch, gac,1);
    new_dw->get(diff_coeff, d_scalar->diffusionCoefLabel,  indx, patch, gac,1);
    new_dw->allocateAndPut(f_src, d_scalar->scalar_source_CCLabel,
                                                           indx, patch);
                                                      
    new_dw->getModifiable(mom_src,    mi->momentum_source_CCLabel,indx,patch);
    new_dw->getModifiable(eng_src,    mi->energy_source_CCLabel,  indx,patch);


    //__________________________________
    // rho=1/(f/rho_fuel+(1-f)/rho_air)
    double fuzzyOne = 1.0 + 1e-10;
    double fuzzyZero = 0.0 - 1e10;
    int     numCells = 0, sum = 0;
    double f_stoic= d_scalar->f_stoic;
    //__________________________________   
    for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
      IntVector c = *iter;
      double mass_old = rho_CC_old[c]*volume;
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
      throw InvalidValue(warn.str());
    }         
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
  }
}
//__________________________________      
void SimpleRxn::scheduleComputeStableTimestep(SchedulerP&,
                                      const LevelP&,
                                      const ModelInfo*)
{
  // None necessary...
}
