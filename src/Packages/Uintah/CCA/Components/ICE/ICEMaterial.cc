//  ICEMaterial.cc
#include <Packages/Uintah/CCA/Components/ICE/ICE.h>
#include <Packages/Uintah/CCA/Components/ICE/ICEMaterial.h>
#include <Packages/Uintah/CCA/Components/ICE/EOS/EquationOfState.h>
#include <Packages/Uintah/CCA/Components/ICE/PropertyBase.h>
#include <Packages/Uintah/CCA/Components/ICE/Thermo/ThermoInterface.h>
#include <Packages/Uintah/CCA/Components/ICE/Combined/CombinedFactory.h>
#include <Packages/Uintah/CCA/Components/ICE/Thermo/ThermoFactory.h>
#include <Packages/Uintah/CCA/Components/ICE/Thermo/ConstantThermo.h>
#include <Packages/Uintah/CCA/Components/ICE/EOS/EquationOfStateFactory.h>
#include <Core/Geometry/IntVector.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Packages/Uintah/Core/Grid/Variables/VarLabel.h>
#include <Packages/Uintah/CCA/Components/ICE/GeometryObject2.h>
#include <Packages/Uintah/Core/GeometryPiece/GeometryPieceFactory.h>
#include <Packages/Uintah/Core/GeometryPiece/UnionGeometryPiece.h>
#include <Packages/Uintah/Core/Exceptions/ParameterNotFound.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <iostream>

#define d_TINY_RHO 1.0e-12 // also defined ICE.cc and MPMMaterial.cc 

using namespace std;
using namespace Uintah;
using namespace SCIRun;

 // Constructor
ICEMaterial::ICEMaterial(ProblemSpecP& ps, ModelSetup* setup): Material(ps)
{
  // Follow the layout of the input file
  // Steps:
  // 1a. Look to see if there is a combined EOS/Thermo (and possibly transport) model.
  // 1b. Determine the type of EOS and create it.
  // 1c. Determine the thermodynamic model and create it.
  // 1d. Determine the transport properties model and create it.
  // 2.  Get the general properties of the material such as
  //     density, thermal_conductivity, specific_heat.
  // 3.  Assign the velocity field.

  // Step 1a -- look for a combined model.  This allows a single subcomponent
  //   to operate as an equation of state and thermo interface (and possibly a
  //   a transport interface.
  d_combined = CombinedFactory::create(ps, setup, this);
  if(d_combined){
    // If the properties object implements the particular model, these casts
    // will succeed, otherwise they will fail and the model will be specified
    // below.
    d_eos = dynamic_cast<EquationOfState*>(d_combined);
    d_thermo = dynamic_cast<ThermoInterface*>(d_combined);
    //d_transport = dynamic_cast<TransportInterface*>(d_combined);
  } else {
    d_eos = 0;
    d_thermo = 0;
  }

  // Step 1b -- create the equation of state if necessary
  if(!d_eos){
    d_eos = EquationOfStateFactory::create(ps, this);
    if(!d_eos) {
      throw ParameterNotFound("ICE: No EOS specified", __FILE__, __LINE__);
    }
  }
   
  // Step 1c -- create the thermo model if necessary
  if(!d_thermo){ // The thermo model may be from a "combined" model
    d_thermo = ThermoFactory::create(ps, setup, this);
    if(!d_thermo){
      // default to constant thermo
      d_thermo = new ConstantThermo(ps, setup, this);
    }
  }

  // Step 1d -- create the transport model if necessary
  // Not finished

   // Step 2 -- get the general material properties
#if 0
   ps->require("thermal_conductivity",d_thermalConductivity);
   ps->require("specific_heat",d_specificHeat);
   ps->require("speed_of_sound",d_speed_of_sound);
   ps->require("gamma",d_gamma);
#endif
   ps->require("dynamic_viscosity",d_viscosity);
   
   d_isSurroundingMatl = false;
   ps->get("isSurroundingMatl",d_isSurroundingMatl);

   d_includeFlowWork = true;
   ps->get("includeFlowWork",d_includeFlowWork);

   // Step 3 -- Loop through all of the pieces in this geometry object
   int piece_num = 0;
   for (ProblemSpecP geom_obj_ps = ps->findBlock("geom_object");
        geom_obj_ps != 0;
        geom_obj_ps = geom_obj_ps->findNextBlock("geom_object") ) {

      vector<GeometryPiece*> pieces;
      GeometryPieceFactory::create(geom_obj_ps, pieces);

      GeometryPiece* mainpiece;
      if(pieces.size() == 0){
         throw ParameterNotFound("No piece specified in geom_object", __FILE__, __LINE__);
      } else if(pieces.size() > 1){
         mainpiece = scinew UnionGeometryPiece(pieces);
      } else {
         mainpiece = pieces[0];
      }

      piece_num++;
      d_geom_objs.push_back(scinew GeometryObject2(this,mainpiece,geom_obj_ps));
   }
   lb = scinew ICELabel();
}
 // Destructor
ICEMaterial::~ICEMaterial()
{
  // Be careful not to delete the combined models twice
  if(d_eos != dynamic_cast<EquationOfState*>(d_combined)){
    delete d_eos;
  }
  if(d_thermo != dynamic_cast<ThermoInterface*>(d_combined)){
    delete d_thermo;
  }

  //if(d_transport != dynamic_cast<TransportInterface*>(d_properties)){
  //delete d_transport;
  //}
  if(d_combined)
    delete d_combined;
  for (int i = 0; i< (int)d_geom_objs.size(); i++) {
    delete d_geom_objs[i];
  }
}

EquationOfState * ICEMaterial::getEOS() const
{
  return d_eos;
}

ThermoInterface* ICEMaterial::getThermo() const
{
  return d_thermo;
}

double ICEMaterial::getViscosity() const
{
  return d_viscosity;
}

bool ICEMaterial::isSurroundingMatl() const
{
  return d_isSurroundingMatl;
}

/* --------------------------------------------------------------------- 
 Function~  ICEMaterial::initializeCells--
 Purpose~ Initialize material dependent variables 
 Notes:  This is a tricky little routine.  ICE needs rho_micro, Temp_CC
 speedSound defined everywhere for all materials even if the mass is 0.0.
 
 We need to accomidate the following scenarios where the number designates
 a material and * represents a high temperature
 ____________________           ____________________
 | 1  | 1  | 1  | 1  |          | 1  | 1  | 1  | 1  |
 |____|____|____|____|          |____|____|____|____|
 | 1  | 1  | 1* | 1  |          | 1  | 1  | 2* | 1  |
 |____|____|____|____|          |____|____|____|____|
 | 1  | 1  | 1  | 1  |          | 1  | 1  | 1  | 1  |
 |____|____|____|____|          |____|____|____|____|=
_____________________________________________________________________*/
void ICEMaterial::initializeCells(CCVariable<double>& rho_micro,
                                  CCVariable<double>& rho_CC,
                                  CCVariable<double>& temp,
                                  CCVariable<double>& vol_frac_CC,
                                  CCVariable<Vector>& vel_CC,
                                  CCVariable<double>& press_CC,
                                  int numMatls,
                                  const Patch* patch,
                                  DataWarehouse* new_dw)
{
  CCVariable<int> IveBeenHere;
  new_dw->allocateTemporary(IveBeenHere, patch);
  
  // Zero the arrays so they don't get wacky values
  vel_CC.initialize(Vector(0.,0.,0.));
  rho_micro.initialize(0.);
  rho_CC.initialize(0.);
  temp.initialize(0.);
  vol_frac_CC.initialize(0.);
  IveBeenHere.initialize(-9);

  for(int obj=0; obj<(int)d_geom_objs.size(); obj++){
   GeometryPiece* piece = d_geom_objs[obj]->getPiece();
   // Box b1 = piece->getBoundingBox();
   // Box b2 = patch->getBox();
    
   IntVector ppc = d_geom_objs[obj]->getNumParticlesPerCell();
   Vector dxpp = patch->dCell()/ppc;
   Vector dcorner = dxpp*0.5;
   double totalppc = ppc.x()*ppc.y()*ppc.z();

  for(CellIterator iter = patch->getExtraCellIterator(); !iter.done(); iter++){
 
     Point lower = patch->nodePosition(*iter) + dcorner;
     int count = 0;
     for(int ix=0;ix < ppc.x(); ix++){
       for(int iy=0;iy < ppc.y(); iy++){
        for(int iz=0;iz < ppc.z(); iz++){
          IntVector idx(ix, iy, iz);
          Point p = lower + dxpp*idx;
          if(piece->inside(p))
            count++;
        }
       }
     }
  //__________________________________
  // For single materials with more than one object 
      if(numMatls == 1)  {
        if ( count > 0 ) {
          vol_frac_CC[*iter]= 1.0;
          press_CC[*iter]   = d_geom_objs[obj]->getInitialPressure();
          vel_CC[*iter]     = d_geom_objs[obj]->getInitialVelocity();
          rho_micro[*iter]  = d_geom_objs[obj]->getInitialDensity();
          rho_CC[*iter]     = rho_micro[*iter] + d_TINY_RHO*rho_micro[*iter];
          temp[*iter]       = d_geom_objs[obj]->getInitialTemperature();
          IveBeenHere[*iter]= 1;
        }
      }   
      if (numMatls > 1 ) {
        vol_frac_CC[*iter]+= count/totalppc;       
        if(IveBeenHere[*iter] == -9){
          // This cell hasn't been hit for this matl yet so set values
          // to ensure that everything is set to something everywhere
          vel_CC[*iter]     = d_geom_objs[obj]->getInitialVelocity();
          rho_micro[*iter]  = d_geom_objs[obj]->getInitialDensity();
          rho_CC[*iter]     = rho_micro[*iter] * vol_frac_CC[*iter] +
                            d_TINY_RHO*rho_micro[*iter];
          temp[*iter]       = d_geom_objs[obj]->getInitialTemperature();
          IveBeenHere[*iter]= obj; 
        }
        if(IveBeenHere[*iter] != -9 && count > 0){
          // This cell HAS been hit but another object has values to
          // override it, possibly in a cell that was just set by default
          // in the above section.
          press_CC[*iter]   = d_geom_objs[obj]->getInitialPressure();
          vel_CC[*iter]     = d_geom_objs[obj]->getInitialVelocity();
          rho_micro[*iter]  = d_geom_objs[obj]->getInitialDensity();
          rho_CC[*iter]     = rho_micro[*iter] * vol_frac_CC[*iter] +
                            d_TINY_RHO*rho_micro[*iter];
          temp[*iter]       = d_geom_objs[obj]->getInitialTemperature();
          IveBeenHere[*iter]= obj; 
        }
        if(IveBeenHere[*iter] != -9 && count == 0){
          // This cell has been initialized, the current object doesn't
          // occupy this cell, so don't screw with it.  All bases are
          // covered.
        }
      }    
    }  // Loop over domain
  }  // Loop over geom_objects
}

