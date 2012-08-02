/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


#ifndef __ICE_MATERIAL_H__
#define __ICE_MATERIAL_H__

// Do not EVER put a #include for anything in CCA/Components in here.
// Ask steve for a better way

#include <CCA/Ports/DataWarehouseP.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Grid/Material.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <CCA/Components/ICE/SpecificHeatModel/SpecificHeat.h>
#include <Core/Geometry/Vector.h>

#include <vector>

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
 
 class ICEMaterial : public Material {
 public:
   ICEMaterial(ProblemSpecP&);
   
   ~ICEMaterial();

   virtual ProblemSpecP outputProblemSpec(ProblemSpecP& ps);
   
   //////////
   // Return correct EOS model pointer for this material
   EquationOfState* getEOS() const;

   // Get the associated specific heat model.  
   // If there is none specified, this will return a null (0) pointer
   SpecificHeat* getSpecificHeatModel() const;
   
   //for HeatConductionModel
   double getGamma() const;
   double getViscosity() const;
   double getSpeedOfSound() const;
   bool   isSurroundingMatl() const;
   bool getIncludeFlowWork() const;
   double getSpecificHeat() const;
   double getThermalConductivity() const;
   double getInitialDensity() const;
   double getTinyRho() const;

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
   SpecificHeat    *d_cv;  // Specific heat model
   double d_viscosity;
   double d_gamma;
   bool d_isSurroundingMatl; // defines which matl is the background matl.
   bool d_includeFlowWork;
   double d_specificHeat;
   double d_thermalConductivity;
   double d_tiny_rho;

   std::vector<GeometryObject*> d_geom_objs;

   ICELabel* lb;
   
   // Prevent copying of this class
   // copy constructor
   ICEMaterial(const ICEMaterial &icem);
   ICEMaterial& operator=(const ICEMaterial &icem);        
   
 };
} // End namespace Uintah

#endif // __ICE_MATERIAL_H__





