//  ICEMaterial.cc

#include "ICEMaterial.h"
#include <Core/Geometry/IntVector.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <Packages/Uintah/Core/Grid/PerPatch.h>
#include <Packages/Uintah/CCA/Components/ICE/GeometryObject2.h>
#include <Packages/Uintah/Core/Grid/GeometryPieceFactory.h>
#include <Packages/Uintah/Core/Grid/UnionGeometryPiece.h>
#include <Packages/Uintah/Core/Exceptions/ParameterNotFound.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <iostream>
#include <Packages/Uintah/CCA/Components/ICE/EOS/EquationOfStateFactory.h>

#define YOU_CANNOT_DO_THIS 1.0e-100  // d_SMALL_NUM

using namespace std;
using namespace Uintah;
using namespace SCIRun;

ICEMaterial::ICEMaterial(ProblemSpecP& ps)
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

   // Step 2 -- get the general material properties

   ps->require("density",d_density);
   ps->require("thermal_conductivity",d_thermalConductivity);
   ps->require("specific_heat",d_specificHeat);
   ps->require("viscosity",d_viscosity);
   ps->require("speed_of_sound",d_speed_of_sound);
   ps->require("gamma",d_gamma);

   cout << "Density = " << d_density << endl;
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
  delete lb;
}

EquationOfState * ICEMaterial::getEOS() const
{
  // Return the pointer to the constitutive model associated
  // with this material

  return d_eos;
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

void ICEMaterial::initializeCells(CCVariable<double>& rho_micro,
				  CCVariable<double>& rho_CC,
				  CCVariable<double>& temp,
				  CCVariable<double>& cv,
				  CCVariable<double>& speedSound,
				  CCVariable<double>& visc_CC,
				  CCVariable<double>& vol_frac_CC,
				  CCVariable<Vector>& vel_CC,
				  const Patch* patch,
				  DataWarehouseP& new_dw)
{
  // Zero the arrays so they don't get wacky values
  vel_CC.initialize(Vector(0.,0.,0.));
  rho_micro.initialize(0.);
  rho_CC.initialize(0.);
  temp.initialize(0.);
  vol_frac_CC.initialize(0.);
  speedSound.initialize(0.);
  visc_CC.initialize(0.);
  cv.initialize(0.);

  for(int i=0; i<(int)d_geom_objs.size(); i++){
   GeometryPiece* piece = d_geom_objs[i]->getPiece();
   Box b1 = piece->getBoundingBox();
   Box b2 = patch->getBox();
   Box b = b1.intersect(b2);
   
   if(b.degenerate())
      cerr << "b.degenerate" << endl;

   IntVector ppc = d_geom_objs[i]->getNumParticlesPerCell();
   Vector dxpp = patch->dCell()/ppc;
   Vector dcorner = dxpp*0.5;
   double totalppc = ppc.x()*ppc.y()*ppc.z();

   for(CellIterator iter = patch->getExtraCellIterator(b); !iter.done(); 
       iter++){
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
     
    if( count > 0)
    {
       vol_frac_CC[*iter]= count/totalppc;
    }
       vel_CC[*iter]     = d_geom_objs[i]->getInitialVelocity();
       speedSound[*iter] = d_speed_of_sound;
       visc_CC[*iter]    = d_viscosity;
       temp[*iter]       = d_geom_objs[i]->getInitialTemperature();
       cv[*iter]         = d_specificHeat;
       rho_micro[*iter]  = d_density;
       rho_CC[*iter]     = d_density*vol_frac_CC[*iter] + YOU_CANNOT_DO_THIS;
   }
  }
}

