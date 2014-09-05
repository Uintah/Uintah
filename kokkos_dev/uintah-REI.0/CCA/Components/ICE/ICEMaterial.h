#ifndef __ICE_MATERIAL_H__
#define __ICE_MATERIAL_H__

// Do not EVER put a #include for anything in CCA/Components in here.
// Ask steve for a better way

#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/Grid/Material.h>
#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>

#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

namespace SCIRun {
  class Point;
  class Vector;
}

#include <Packages/Uintah/CCA/Components/ICE/share.h>
namespace Uintah {
  using namespace SCIRun;
  class ICELabel;
  class EquationOfState;
  class GeometryObject;
 
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
 
 class SCISHARE ICEMaterial : public Material {
 public:
   ICEMaterial(ProblemSpecP&);
   
   ~ICEMaterial();

   virtual ProblemSpecP outputProblemSpec(ProblemSpecP& ps);
   
   //////////
   // Return correct EOS model pointer for this material
   EquationOfState* getEOS() const;
   
   //for HeatConductionModel
   double getGamma() const;
   double getViscosity() const;
   double getSpeedOfSound() const;
   bool   isSurroundingMatl() const;
   
   void initializeCells(CCVariable<double>& rhom,
                     CCVariable<double>& rhC,
                     CCVariable<double>& temp, 
                     CCVariable<double>& ss,
                     CCVariable<double>& volf,  CCVariable<Vector>& vCC,
                     CCVariable<double>& press,
                     int numMatls,
                     const Patch* patch, DataWarehouse* new_dw);
   
 private:
   
   // Specific constitutive model associated with this material
   EquationOfState *d_eos;
   double d_viscosity;
   double d_gamma;
   bool d_isSurroundingMatl; // defines which matl is the background matl.
   
   std::vector<GeometryObject*> d_geom_objs;

   ICELabel* lb;
   
   // Prevent copying of this class
   // copy constructor
   ICEMaterial(const ICEMaterial &icem);
   ICEMaterial& operator=(const ICEMaterial &icem);        
   
 };
} // End namespace Uintah

#endif // __ICE_MATERIAL_H__





