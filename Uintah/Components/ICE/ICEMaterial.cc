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
      // Step 4 -- Assign the boundary conditions to the object


      // Step 5 -- Assign the velocity field
      int vf;
      ps->require("velocity_field",vf);
      setVFIndex(vf);
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
				  CCVariable<double>& uvel_CC,
				  CCVariable<double>& vvel_CC,
				  CCVariable<double>& wvel_CC,
				  const Patch* patch,
				  DataWarehouseP& new_dw)
{
  double volume_fraction = 1.0;

  for(int i=0; i<(int)d_geom_objs.size(); i++){
   GeometryPiece* piece = d_geom_objs[i]->getPiece();
   Box b1 = piece->getBoundingBox();
   Box b2 = patch->getBox();
   Box b = b1.intersect(b2);
   if(b.degenerate())
      cerr << "b.degenerate" << endl;

    // Set the initial conditions:
    uvel_CC.initialize(d_geom_objs[i]->getInitialVelocity().x());
    vvel_CC.initialize(d_geom_objs[i]->getInitialVelocity().y());
    wvel_CC.initialize(d_geom_objs[i]->getInitialVelocity().z());
    rho_micro.initialize(d_density);
    rho_CC.initialize(d_density*volume_fraction);
    temp.initialize(d_geom_objs[i]->getInitialTemperature());
    vol_frac_CC.initialize(volume_fraction);
    speedSound.initialize(d_speed_of_sound);
    visc_CC.initialize(d_viscosity);
    cv.initialize(d_specificHeat);
  }
}

// $Log$
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
