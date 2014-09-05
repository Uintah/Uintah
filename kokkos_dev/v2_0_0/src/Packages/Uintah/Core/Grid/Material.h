#ifndef __MATERIAL_H__
#define __MATERIAL_H__

#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>
#include <Packages/Uintah/Core/Grid/ComputeSet.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>


namespace Uintah {
  class Burn;

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
     enum RX_Prod {reactant,product,none};

     Material(ProblemSpecP& ps);
     Material();
      
     virtual ~Material();
      
      //////////
      // Return index associated with this material's
      // location in the data warehouse
      int getDWIndex() const;
      
      //////////
      // Return index associated with this material's
      // velocity field
      int getVFIndex() const;

      virtual Burn* getBurnModel();

      void setDWIndex(int);
      void setVFIndex(int);
      RX_Prod getRxProduct();

     const MaterialSubset* thisMaterial() const {
       return thismatl;
     }

     bool hasName() const {
       return haveName;
     }
     std::string getName() const {
       return name;
     }
   protected:

      Burn* d_burn;
      // Index associated with this material's spot in the DW
      int d_dwindex;
      // Index associated with this material's velocity field
      int d_vfindex;
     MaterialSubset* thismatl;

     RX_Prod d_rx_prod;
   private:

     bool haveName;
     std::string name;
      
      Material(const Material &mat);
      Material& operator=(const Material &mat);
   };
} // End namespace Uintah

#endif // __MATERIAL_H__
