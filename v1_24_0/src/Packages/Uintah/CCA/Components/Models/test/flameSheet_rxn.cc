#include <Packages/Uintah/CCA/Components/ICE/ICEMaterial.h>
#include <Packages/Uintah/CCA/Components/ICE/Diffusion.h>
#include <Packages/Uintah/CCA/Components/Models/test/flameSheet_rxn.h>
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
  mymatls = 0;
}
//__________________________________
flameSheet_rxn::~flameSheet_rxn()
{
  if(mymatls && mymatls->removeReference())
    delete mymatls;
  for(vector<Scalar*>::iterator iter = scalars.begin();
      iter != scalars.end(); iter++){
    Scalar* scalar = *iter;
    VarLabel::destroy(scalar->scalar_CCLabel);
    VarLabel::destroy(scalar->scalar_source_CCLabel);
    for(vector<Region*>::iterator iter = scalar->regions.begin();
	iter != scalar->regions.end(); iter++){
      Region* region = *iter;
      delete region->piece;
      delete region;
    }
  }
}
//__________________________________
//
flameSheet_rxn::Region::Region(GeometryPiece* piece, ProblemSpecP& ps)
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
  sharedState = in_state;
  matl = sharedState->parseAndLookupMaterial(params, "material");

  vector<int> m(1);
  m[0] = matl->getDWIndex();
  mymatls = new MaterialSet();
  mymatls->addAll(m);
  mymatls->addReference();

  // determine the specific heat of that matl.
  Material* matl = sharedState->getMaterial( m[0] );
  ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
  if (ice_matl){
    d_cp = ice_matl->getSpecificHeat();
  }   
  
  Scalar* scalar = new Scalar();
  scalar->index = 0;
  scalar->name  = "f";
  string Sname     = "scalar-"+scalar->name;
  string Ssrc_name = "scalarSource-"+scalar->name;
  scalar->scalar_CCLabel = 
      VarLabel::create(Sname,     CCVariable<double>::getTypeDescription());
  scalar->scalar_source_CCLabel = 
      VarLabel::create(Ssrc_name, CCVariable<double>::getTypeDescription());

  setup->registerTransportedVariable(mymatls->getSubset(0),
				    scalar->scalar_CCLabel,
				    scalar->scalar_source_CCLabel);
  scalars.push_back(scalar);
  names[scalar->name] = scalar;

  if(scalars.size() == 0) {
    throw ProblemSetupException("flameSheet_rxn: no scalar specifed!");
  }

  for (ProblemSpecP child = params->findBlock("scalar"); child != 0;
       child = child->findNextBlock("scalar")) {
    string name;
    child->getAttribute("name", name);
    map<string, Scalar*>::iterator iter = names.find(name);
    if(iter == names.end()) {
      throw ProblemSetupException("Scalar "+name+" species not found");
    }
    //__________________________________
    //  reaction constants
    ProblemSpecP react_ps = child->findBlock("reaction_constants");
    if(!react_ps) {
      throw ProblemSetupException("Cannot find reaction_constants tag");
    }
    
    react_ps->getWithDefault("f_stoichometric",       d_f_stoic,       -9);  
    react_ps->getWithDefault("delta_H_combustion",    d_del_h_comb,    -9);  
    react_ps->getWithDefault("oxidizer_temp_infinity",d_T_oxidizer_inf,-9);
    react_ps->getWithDefault("initial_fuel_temp",     d_T_fuel_init,   -9);    
    react_ps->getWithDefault("diffusivity",           d_diffusivity,   -9);
    react_ps->getWithDefault("smear_initialDistribution_knob",       
                              d_smear_initialDistribution_knob,       0);   
    if( d_f_stoic == -9        ||  d_del_h_comb == -9 ||    // bulletproofing
        d_T_oxidizer_inf == -9 ||  d_T_fuel_init == -9 ) {
      ostringstream warn;
      warn << " ERROR FlameSheet_rxn: Input variable(s) not specified \n" 
           << "\n f_stoichometric        "<< d_f_stoic
           << "\n delta_H_combustion     "<< d_del_h_comb
           << "\n oxidizer_temp_infinity "<< d_T_oxidizer_inf
           << "\n fuel_temp_init         "<< d_T_fuel_init 
           << "\n diffusivity            "<< d_diffusivity<< endl;
      throw ProblemSetupException(warn.str());
    }
    
    //__________________________________
    //  geom objects
    Scalar* scalar = iter->second;
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

      scalar->regions.push_back(scinew Region(mainpiece, geom_obj_ps));
    }
    if(scalar->regions.size() == 0) {
      throw ProblemSetupException("Variable: "+scalar->name+" does not have any initial value regions");
    }
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
  for(vector<Scalar*>::iterator iter = scalars.begin();
      iter != scalars.end(); iter++){
    Scalar* scalar = *iter;
    t->computes(scalar->scalar_CCLabel);
  }
  sched->addTask(t, level->eachPatch(), mymatls);
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
    
    for(int m=0;m<matls->size();m++){
      int matl = matls->get(m);
      CCVariable<double>  var;
      
      for(vector<Scalar*>::iterator iter = scalars.begin();
	  iter != scalars.end(); iter++){
	 Scalar* scalar = *iter;

	 new_dw->allocateAndPut(var, scalar->scalar_CCLabel, matl, patch);
	 var.initialize(0);

	 for(vector<Region*>::iterator iter = scalar->regions.begin();
	      iter != scalar->regions.end(); iter++){
	   Region* region = *iter;
          
	   for(CellIterator iter = patch->getExtraCellIterator();
	       !iter.done(); iter++){
            IntVector c = *iter;
	     Point p = patch->cellPosition(c);            
	     if(region->piece->inside(p)) {
	       var[c] = region->initialScalar;
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
          constCCVariable<double> placeHolder;
          
          scalarDiffusionOperator(new_dw, patch, use_vol_frac,
                                  placeHolder, placeHolder,  var,
                                  var, FakeDiffusivity, fakedelT); 
        }  // diffusion loop
      } // over scalars
    } // Over matls
  }  // patches
}
//__________________________________      
void flameSheet_rxn::scheduleComputeStableTimestep(SchedulerP&,
					   const LevelP&,
					   const ModelInfo*)
{
  // None necessary...
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
  t->modifies(mi->energy_source_CCLabel);
  t->requires(Task::OldDW, mi->density_CCLabel,     gn);
  t->requires(Task::OldDW, mi->temperature_CCLabel, gn);
  t->requires(Task::NewDW, mi->specific_heatLabel,  gn);
  t->requires(Task::NewDW, mi->gammaLabel,          gn);
  //t->requires(Task::OldDW, mi->delT_Label);   AMR
  
  for(vector<Scalar*>::iterator iter = scalars.begin();
      iter != scalars.end(); iter++){
    Scalar* scalar = *iter;
    t->requires(Task::OldDW, scalar->scalar_CCLabel, gac,1);
    t->modifies(scalar->scalar_source_CCLabel);
  }
  sched->addTask(t, level->eachPatch(), mymatls);
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

    Vector dx = patch->dCell();
    double volume = dx.x()*dx.y()*dx.z();
    Ghost::GhostType  gn = Ghost::None;  
    Ghost::GhostType  gac = Ghost::AroundCells;     
    
    for(int m=0;m<matls->size();m++){
      int matl = matls->get(m);
      constCCVariable<double> rho_CC, Temp_CC,f_old, cv, gamma;
      CCVariable<double> energySource, f_src;
      
      double new_f, newTemp;
      double Y_fuel,Y_products;
            
      old_dw->get(rho_CC,      mi->density_CCLabel,     matl, patch, gn, 0);
      old_dw->get(Temp_CC,     mi->temperature_CCLabel, matl, patch, gn, 0);
      new_dw->get(cv,          mi->specific_heatLabel,  matl, patch, gn, 0);
      new_dw->get(gamma,       mi->gammaLabel,          matl, patch, gn, 0);
      new_dw->getModifiable(energySource,   
                               mi->energy_source_CCLabel,matl,patch);
      // transported scalar.                        
      for(vector<Scalar*>::iterator iter = scalars.begin();
	  iter != scalars.end(); iter++){
	 Scalar* scalar = *iter;
        old_dw->get(f_old,            scalar->scalar_CCLabel,       
                                                        matl, patch, gac, 1);
        new_dw->allocateAndPut(f_src, scalar->scalar_source_CCLabel,
                                                        matl, patch, gn, 0);
      }
      
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
      
      //__________________________________   
      for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
	 IntVector c = *iter;
        double cp = gamma[c] * cv[c];
	 double mass = rho_CC[c]*volume;
	 double f = f_old[c];
	 double oldTemp  = Temp_CC[c];
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
        
        new_f =Y_fuel + Y_products/(1.0 + nu);        // eqs 7.54         
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
        throw InvalidValue(warn.str());
      }         
      //__________________________________
      //  Tack on diffusion
      if(d_diffusivity != 0.0){ 
        bool use_vol_frac = false; // don't include vol_frac in diffusion calc.
        constCCVariable<double> placeHolder;
        /*`==========TESTING==========*/    
        // this needs to be changed
        CCVariable<double> diff_coeff;
        new_dw->allocateTemporary(diff_coeff, patch,Ghost::AroundCells, 1);
        diff_coeff.initialize(d_diffusivity);    
        /*==========TESTING==========`*/
        scalarDiffusionOperator(new_dw, patch, use_vol_frac,
                                placeHolder, placeHolder,  f_old,
                                f_src, diff_coeff, delT);
      }  // diffusivity > 0 
    }  // matl loop
  }
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
