//  ICEMaterial.cc
//

#include "ICEMaterial.h"
#include <SCICore/Geometry/IntVector.h>
#include <Uintah/Grid/Patch.h>
#include <Uintah/Grid/CellIterator.h>
#include <Uintah/Grid/VarLabel.h>
#include <Uintah/Grid/PerPatch.h>
#include <Uintah/Components/ICE/GeometryObject2.h>
#include <Uintah/Grid/GeometryPieceFactory.h>
#include <Uintah/Grid/UnionGeometryPiece.h>
#include <Uintah/Exceptions/ParameterNotFound.h>
#include <Uintah/Interface/DataWarehouse.h>
#include <iostream>
#include <Uintah/Components/ICE/EOS/EquationOfStateFactory.h>


#define d_SMALL_NUM 1.0e-100
using namespace std;
using namespace Uintah::ICESpace;
using namespace Uintah;
using namespace SCICore::Geometry;

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
				  const Patch* patch,DataWarehouseP& new_dw)
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
       rho_CC[*iter]     = d_density*vol_frac_CC[*iter] + d_SMALL_NUM;
   }
  }
}

// $Log$
// Revision 1.13  2001/01/11 22:38:50  guilkey
// Get rid of requirement for velocity_field flag from ps.
//
// Revision 1.12  2001/01/11 00:22:06  harman
// -remove if(count) conditional around, vel, cv, temp, cv, speedSound,visc_CC
//  these need to be defined in everycell for every mat.
// - added d_SMALL_NUM to rho_CC computation.
//
// Revision 1.11  2001/01/05 16:34:10  jas
// Changed over uvel_CC, vvel_CC, wvel_CC to a CCVariable<Vector> in all steps
// where CC velocities are used.
//
// Revision 1.10  2000/12/18 23:25:55  jas
// 2d ice works for simple advection.
//
// Revision 1.9  2000/12/05 15:45:30  jas
// Now using SFC{X,Y,Z} data types.  Fixed some small bugs and things appear
// to be working up to the middle of step 2.
//
// Revision 1.8  2000/11/28 03:50:28  jas
// Added {X,Y,Z}FCVariables.  Things still don't work yet!
//
// Revision 1.7  2000/11/23 00:45:45  guilkey
// Finished changing the way initialization of the problem was done to allow
// for different regions of the domain to be easily initialized with different
// materials and/or initial values.
//
// Revision 1.6  2000/11/22 01:28:05  guilkey
// Changed the way initial conditions are set.  GeometryObjects are created
// to fill the volume of the domain.  Each object has appropriate initial
// conditions associated with it.  ICEMaterial now has an initializeCells
// method, which for now just does what was previously done with the
// initial condition stuct d_ic.  This will be extended to allow regions of
// the domain to be initialized with different materials.  Sorry for the
// lame GeometryObject2, this could be changed to ICEGeometryObject or
// something.
//
// Revision 1.5  2000/10/27 23:41:01  jas
// Added more material constants and some debugging output.
//
// Revision 1.4  2000/10/06 04:05:18  jas
// Move files into EOS directory.
//
// Revision 1.3  2000/10/05 04:26:48  guilkey
// Added code for part of the EOS evaluation.
//
// Revision 1.2  2000/10/04 20:17:52  jas
// Change namespace ICE to ICESpace.
//
// Revision 1.1  2000/10/04 19:26:14  guilkey
// Initial commit of some classes to help mainline ICE.
//
