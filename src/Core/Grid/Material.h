/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
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


#ifndef __MATERIAL_H__
#define __MATERIAL_H__

#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>
#include <Packages/Uintah/Core/Grid/Variables/ComputeSet.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/Grid/SimulationStateP.h>


#include <Packages/Uintah/Core/Grid/uintahshare.h>
namespace Uintah {

/**************************************

CLASS
   Material

   Short description...

GENERAL INFORMATION

   Material.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)

   Copyright (C) 2000 SCI Group

KEYWORDS
   Material

DESCRIPTION
   Long description...

WARNING

****************************************/

//using ::Grid::Patch;
//using ::Interface::DataWarehouseP;


   class UINTAHSHARE Material {
   public:
     Material(ProblemSpecP& ps);
     Material();
      
     virtual ~Material();
      
     virtual ProblemSpecP outputProblemSpec(ProblemSpecP& ps);

      //////////
      // Return index associated with this material's
      // location in the data warehouse
      int getDWIndex() const;
      
      //////////
      // Return index associated with this material's
      // velocity field
      int getVFIndex() const;

      void setDWIndex(int);
      void setVFIndex(int);

     const MaterialSubset* thisMaterial() const {
       return thismatl;
     }
     
     virtual void registerParticleState(SimulationState* ss);

     double getThermalConductivity() const;
     double getSpecificHeat() const;
     double getHeatTransferCoefficient() const;
     bool getIncludeFlowWork() const;
     bool hasName() const {
       return haveName;
     }
     std::string getName() const {
       return name;
     }
   protected:

      // Index associated with this material's spot in the DW
      int d_dwindex;
      // Index associated with this material's velocity field
      int d_vfindex;
      MaterialSubset* thismatl;
      double d_thermalConductivity;
      double d_specificHeat;
      double d_heatTransferCoefficient;
      bool d_includeFlowWork;

   private:

     bool haveName;
     std::string name;
      
     Material(const Material &mat);
     Material& operator=(const Material &mat);
   };
} // End namespace Uintah

#endif // __MATERIAL_H__
