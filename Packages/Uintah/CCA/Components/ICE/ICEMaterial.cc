//  ICEMaterial.cc
#include <Packages/Uintah/CCA/Components/ICE/ICE.h>
#include <Packages/Uintah/CCA/Components/ICE/ICEMaterial.h>
#include <Core/Geometry/IntVector.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <Packages/Uintah/CCA/Components/ICE/GeometryObject2.h>
#include <Packages/Uintah/Core/Grid/GeomPiece/GeometryPieceFactory.h>
#include <Packages/Uintah/Core/Grid/GeomPiece/UnionGeometryPiece.h>
#include <Packages/Uintah/Core/Exceptions/ParameterNotFound.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <iostream>
#include <Packages/Uintah/CCA/Components/ICE/EOS/EquationOfStateFactory.h>
#include <Packages/Uintah/CCA/Components/HETransformation/BurnFactory.h>
#include <Packages/Uintah/CCA/Components/HETransformation/Burn.h>
#include <Core/Util/NotFinished.h>

#define d_TINY_RHO 1.0e-12 // also defined ICE.cc and MPMMaterial.cc 

using namespace std;
using namespace Uintah;
using namespace SCIRun;

ICEMaterial::ICEMaterial(ProblemSpecP& ps): Material(ps)
{

  // Constructor

  // Follow the layout of the input file
  // Steps:
  // 1.  Determine the type of EOS and create it.
  // 2.  Get the general properties of the material such as
  //     density, thermal_conductivity, specific_heat.
  // 3.  Assign the velocity field.

  // Step 1 -- create the constitutive gmodel.
   d_eos = EquationOfStateFactory::create(ps);
   if(!d_eos)
      throw ParameterNotFound("No EOS");
   
   d_burn = BurnFactory::create(ps);


   // Step 2 -- get the general material properties
   ps->require("thermal_conductivity",d_thermalConductivity);
   ps->require("specific_heat",d_specificHeat);
   ps->require("viscosity",d_viscosity);
   ps->require("speed_of_sound",d_speed_of_sound);
   ps->require("gamma",d_gamma);

   // Step 3 -- Loop through all of the pieces in this geometry object
   int piece_num = 0;
   for (ProblemSpecP geom_obj_ps = ps->findBlock("geom_object");
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

      piece_num++;
      d_geom_objs.push_back(scinew GeometryObject2(this,mainpiece,geom_obj_ps));
   }

   lb = scinew ICELabel();
}

ICEMaterial::~ICEMaterial()
{
  // Destructor

  delete d_eos;
  delete d_burn;
  delete lb;
  for (int i = 0; i< (int)d_geom_objs.size(); i++) {
       delete d_geom_objs[i];
  }
}

EquationOfState * ICEMaterial::getEOS() const
{
  // Return the pointer to the constitutive model associated
  // with this material

  return d_eos;
}

Burn* ICEMaterial::getBurnModel()
{
  // Return the pointer to the burn model associated
  // with this material

  return d_burn;
}

double ICEMaterial::getThermalConductivity() const
{
  return d_thermalConductivity;
}

double ICEMaterial::getSpecificHeat() const
{
  return d_specificHeat;
}

double ICEMaterial::getHeatTransferCoefficient() const
{
  return d_heatTransferCoefficient;
}

double ICEMaterial::getGamma() const
{
  return d_gamma;
}

double ICEMaterial::getViscosity() const
{
  return d_viscosity;
}

double ICEMaterial::getSpeedOfSound() const
{
  return d_speed_of_sound;
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
                                  CCVariable<double>& speedSound,
                                  CCVariable<double>& vol_frac_CC,
                                  CCVariable<Vector>& vel_CC,
                                  CCVariable<double>& press_CC,
                                  int numMatls,
                              const Patch* patch,DataWarehouse* new_dw)
{
  CCVariable<int> IveBeenHere;
  new_dw->allocateTemporary(IveBeenHere, patch);
  
  // Zero the arrays so they don't get wacky values
  vel_CC.initialize(Vector(0.,0.,0.));
  rho_micro.initialize(0.);
  rho_CC.initialize(0.);
  temp.initialize(0.);
  vol_frac_CC.initialize(0.);
  speedSound.initialize(0.);
  
  for(int obj=0; obj<(int)d_geom_objs.size(); obj++){
   GeometryPiece* piece = d_geom_objs[obj]->getPiece();
   Box b1 = piece->getBoundingBox();
   Box b2 = patch->getBox();
   Box b = b1.intersect(b2);
   
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
     IveBeenHere[*iter]=-9; 
  //__________________________________
  // For single materials with more than one object 
      if(numMatls == 1)  {
        if ( count > 0  && obj == 0) {
          vol_frac_CC[*iter]= 1.0;
          press_CC[*iter]   = d_geom_objs[obj]->getInitialPressure();
          vel_CC[*iter]     = d_geom_objs[obj]->getInitialVelocity();
          rho_micro[*iter]  = d_geom_objs[obj]->getInitialDensity();
          rho_CC[*iter]     = rho_micro[*iter] + d_TINY_RHO*rho_micro[*iter];
          temp[*iter]       = d_geom_objs[obj]->getInitialTemperature();
          speedSound[*iter] = d_speed_of_sound;
          IveBeenHere[*iter]= 1;
        }

        if (count > 0 && obj > 0) {
          vol_frac_CC[*iter]= 1.0;
          press_CC[*iter]   = d_geom_objs[obj]->getInitialPressure();
          vel_CC[*iter]     = d_geom_objs[obj]->getInitialVelocity();
          rho_micro[*iter]  = d_geom_objs[obj]->getInitialDensity();
          rho_CC[*iter]     = rho_micro[*iter] + d_TINY_RHO*rho_micro[*iter];
          temp[*iter]       = d_geom_objs[obj]->getInitialTemperature();
          speedSound[*iter] = d_speed_of_sound;
          IveBeenHere[*iter]= 2;
        } 
      }   
      if (numMatls > 1 ) {
        vol_frac_CC[*iter]= count/totalppc;       
        press_CC[*iter]   = d_geom_objs[obj]->getInitialPressure();
        vel_CC[*iter]     = d_geom_objs[obj]->getInitialVelocity();
        rho_micro[*iter]  = d_geom_objs[obj]->getInitialDensity();
        rho_CC[*iter]     = rho_micro[*iter] * vol_frac_CC[*iter] +
                            d_TINY_RHO*rho_micro[*iter];
        temp[*iter]       = d_geom_objs[obj]->getInitialTemperature();
        speedSound[*iter] = d_speed_of_sound;
        IveBeenHere[*iter]= obj; 
      }    
    }  // Loop over domain
  }  // Loop over geom_objects
}

