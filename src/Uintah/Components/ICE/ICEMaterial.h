#ifndef __ICE_MATERIAL_H__
#define __ICE_MATERIAL_H__

#include <Uintah/Interface/DataWarehouseP.h>
#include <Uintah/Grid/Material.h>
#include <Uintah/Interface/ProblemSpecP.h>
#include <Uintah/Grid/CCVariable.h>
#include <Uintah/Grid/PerPatch.h>
#include <vector>
#include <Uintah/Components/ICE/ICELabel.h>
#include <Uintah/Components/ICE/EOS/EquationOfState.h>
#include <Uintah/Components/ICE/GeometryObject2.h>

namespace SCICore {
   namespace Geometry {
      class Point;
      class Vector;
   }
}

namespace Uintah {
   class Patch;
   class VarLabel;
   namespace ICESpace {
      using SCICore::Geometry::Point;
      using SCICore::Geometry::Vector;
      
/**************************************
     
CLASS
   ICEMaterial

   Short description...

GENERAL INFORMATION

   ICEMaterial.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)

   Copyright (C) 2000 SCI Group

KEYWORDS
   ICE

DESCRIPTION
   Long description...

WARNING

****************************************/

      class ICEMaterial : public Material {
      public:
	 ICEMaterial(ProblemSpecP&);
	 
	 ~ICEMaterial();
	 
	 //////////
	 // Return correct EOS model pointer for this material
	 EquationOfState * getEOS() const;

         //for HeatConductionModel
         double getThermalConductivity() const;
         double getSpecificHeat() const;
         double getHeatTransferCoefficient() const;
         double getGamma() const;
	 double getViscosity() const;
	 double getSpeedOfSound() const;

	 void initializeCells(CCVariable<double>& rhom, CCVariable<double>& rhC,
			      CCVariable<double>& temp, CCVariable<double>& cv,
			      CCVariable<double>& ss,   CCVariable<double>& vsc,
			      CCVariable<double>& volf, CCVariable<double>& uCC,
			      CCVariable<double>& vCC,  CCVariable<double>& wCC,
			      const Patch* patch, DataWarehouseP& new_dw);

      private:

	 // Specific constitutive model associated with this material
	 EquationOfState *d_eos;

	 double d_density;
         double d_thermalConductivity;
         double d_specificHeat;
	 double d_heatTransferCoefficient;
	 double d_speed_of_sound;
	 double d_viscosity;
	 double d_gamma;
         
	 std::vector<GeometryObject2*> d_geom_objs;

	 ICELabel* lb;

	 // Prevent copying of this class
	 // copy constructor
	 ICEMaterial(const ICEMaterial &icem);
	 ICEMaterial& operator=(const ICEMaterial &icem);
      };

} // end namespace ICE
} // end namespace Uintah

#endif // __ICE_MATERIAL_H__

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
