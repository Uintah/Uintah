
#include <Packages/Uintah/CCA/Components/Models/test/flameSheet_rxn.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Material.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Packages/Uintah/Core/Grid/GeomPiece/GeometryPieceFactory.h>
#include <Packages/Uintah/Core/Grid/GeomPiece/UnionGeometryPiece.h>
#include <Packages/Uintah/Core/Exceptions/ParameterNotFound.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/CCA/Components/ICE/ICEMaterial.h>
#include <Core/Containers/StaticArray.h>
#include <Core/Math/MiscMath.h>
#include <iostream>
#include <Core/Util/DebugStream.h>
#include <stdio.h>

using namespace Uintah;
using namespace std;
static DebugStream cout_doing("flameSheet_RXN_DOING_COUT", false);

//______________________________________________________________________
// flame sheet approach for laminar diffusion flames.
// Reference:  "An Introduction to Combustion Concepts and Applications"
//              by Stephen Turns pp 268 - 275
// Assumptions: cp_fuel = cp_oxidizer = cp_products
//              Enthalpies of formation of the oxidizer and products  0.0
//              Thus the enthalpy of formatio of the fuel equal the heat
//              of combustion.

flameSheet_rxn::flameSheet_rxn(const ProcessorGroup* myworld, ProblemSpecP& params)
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
    if( d_f_stoic == -9        ||  d_del_h_comb == -9 ||    // bulletproofing
        d_T_oxidizer_inf == -9 ||  d_T_fuel_init == -9 ) {
      ostringstream warn;
      warn << " ERROR FlameSheet_rxn: Input variable(s) not specified \n" 
           << "\n f_stoichometric        "<< d_f_stoic
           << "\n delta_H_combustion     "<< d_del_h_comb
           << "\n oxidizer_temp_infinity "<< d_T_oxidizer_inf
           << "\n fuel_temp_init         "<< d_T_fuel_init << endl;
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
  cout_doing << "flameSheet_rxn::Initialize " << endl;
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
	   Box b1 = region->piece->getBoundingBox();
	   Box b2 = patch->getBox();
	   Box b = b1.intersect(b2);

	   for(CellIterator iter = patch->getExtraCellIterator();
	       !iter.done(); iter++){

	     Point p = patch->cellPosition(*iter);
	     if(region->piece->inside(p)) {
	       var[*iter] = region->initialScalar;
            }
	   } // Over cells
        } // Over regions
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
void flameSheet_rxn::scheduleMassExchange(SchedulerP&,
				  const LevelP&,
				  const ModelInfo*)
{
  // None required
}
//__________________________________
void flameSheet_rxn::scheduleMomentumAndEnergyExchange(SchedulerP& sched,
					       const LevelP& level,
					       const ModelInfo* mi)
{
  cout_doing << "flameSheet_rxn::react " << endl;
  Task* t = scinew Task("flameSheet_rxn::react",
			this, &flameSheet_rxn::react, mi);
  t->modifies(mi->energy_source_CCLabel);
  t->requires(Task::OldDW, mi->density_CCLabel,     Ghost::None);
  t->requires(Task::OldDW, mi->temperature_CCLabel, Ghost::None);
  
  for(vector<Scalar*>::iterator iter = scalars.begin();
      iter != scalars.end(); iter++){
    Scalar* scalar = *iter;
    t->requires(Task::OldDW, scalar->scalar_CCLabel, Ghost::None);
    t->modifies(scalar->scalar_source_CCLabel);
  }
  sched->addTask(t, level->eachPatch(), mymatls);
}
//______________________________________________________________________
void flameSheet_rxn::react(const ProcessorGroup*, 
		   const PatchSubset* patches,
		   const MaterialSubset* matls,
		   DataWarehouse* old_dw,
		   DataWarehouse* new_dw,
		   const ModelInfo* mi)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    cout_doing << "Doing react on patch "<<patch->getID()<< "\t\t\t\t flameSheet" << endl;    

    Vector dx = patch->dCell();
    double volume = dx.x()*dx.y()*dx.z();
    Ghost::GhostType  gn = Ghost::None;   
    
    for(int m=0;m<matls->size();m++){
      int matl = matls->get(m);
      constCCVariable<double> density, temperature,f_old;
      CCVariable<double> energySource, f_src;
      
      double new_f, newTemp;
      double Y_fuel,Y_products;
            
      old_dw->get(density,     mi->density_CCLabel,     matl, patch, gn, 0);
      old_dw->get(temperature, mi->temperature_CCLabel, matl, patch, gn, 0);
      new_dw->getModifiable(energySource,   
                               mi->energy_source_CCLabel,matl,patch);
      // transported scalar.                        
      for(vector<Scalar*>::iterator iter = scalars.begin();
	  iter != scalars.end(); iter++){
	 Scalar* scalar = *iter;
        old_dw->get(f_old,            scalar->scalar_CCLabel,       
                                                        matl, patch, gn, 0);
        new_dw->allocateAndPut(f_src, scalar->scalar_source_CCLabel,
                                                        matl, patch, gn, 0);
      }
      
      //__________________________________
      //   G R O S S N E S S
      double nu             = (1.0/d_f_stoic) - 1.0;
    //double d_del_h_comb   = 1000.0* 74831.0;    // Enthalpy of combustion J/kg
      double del_h_comb = d_del_h_comb * d_cp/d_f_stoic;
      
      //__________________________________   
      for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
	 IntVector c = *iter;
	 double mass = density[c]*volume;
	 double f = f_old[c];
	 double oldTemp  = temperature[c];

        //__________________________________
        // compute the energy source
        if (d_f_stoic < f && f <= 1.0 ){                  // Inside the flame eqs
          Y_fuel     = (f - d_f_stoic)/(1.0 - d_f_stoic); // 9.43a,b,c & 9.51a
          Y_products = (1 - f)/(1-d_f_stoic);
          
          double tmp = d_f_stoic * del_h_comb/((1.0 - d_f_stoic) * d_cp);
          double A   = f * ( (d_T_fuel_init - d_T_oxidizer_inf) - tmp ); 
          newTemp    =  A + d_T_oxidizer_inf + tmp;
        }
                
        if (d_f_stoic == f ){                          // At the flame surface
          Y_fuel     = 0.0;                            // eqs 9.45a,b,c & 9.51a                         
          Y_products = 1.0;
          
          double A = d_f_stoic *( del_h_comb/d_cp + d_T_fuel_init - d_T_oxidizer_inf);
          newTemp = A + d_T_oxidizer_inf;
        }
      
        if (0 <= f && f < d_f_stoic ){                 //outside the flame
          Y_fuel     = 0.0;                            // eqs 9.46a,b,c & 9.51c
          Y_products = f/d_f_stoic;
          
          double A = f *( (del_h_comb/d_cp) + d_T_fuel_init - d_T_oxidizer_inf);
          newTemp  = A + d_T_oxidizer_inf;
        }       
        new_f =Y_fuel + Y_products/(1.0 + nu);        // eqs 7.54
                
	 double energyx =( newTemp - oldTemp) * d_cp * mass;
        energySource[c] += energyx;
        
	 f_src[c] += new_f - f; 
      }  //iter
    }  // matl loop
  }
}
