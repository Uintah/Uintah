#ifndef __MATERIAL_H__
#define __MATERIAL_H__

#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>
#include <Packages/Uintah/Core/Grid/Variables/ComputeSet.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/Grid/SimulationStateP.h>


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


   class Material {
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
