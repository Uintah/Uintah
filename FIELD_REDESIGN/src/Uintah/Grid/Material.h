#ifndef __MATERIAL_H__
#define __MATERIAL_H__

#include <Uintah/Interface/DataWarehouseP.h>

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

//using Uintah::Grid::Patch;
//using Uintah::Interface::DataWarehouseP;


   class Material {
   public:
      Material() {}
      
      virtual ~Material() {}
      
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
   protected:
      
      // Index associated with this material's spot in the DW
      int d_dwindex;
      // Index associated with this material's velocity field
      int d_vfindex;
      
   private:
      
      Material(const Material &mat);
      Material& operator=(const Material &mat);
   };
   
} // end namespace Uintah

#endif // __MATERIAL_H__

// $Log$
// Revision 1.7  2000/05/30 20:19:29  sparker
// Changed new to scinew to help track down memory leaks
// Changed region to patch
//
// Revision 1.6  2000/04/28 08:11:33  sparker
// ConstitutiveModelFactory should return null on failure
// MPMMaterial checks for failed constitutive model creation
// DWindex and VFindex are now initialized
// Fixed input file to match ConstitutiveModelFactory
//
// Revision 1.5  2000/04/26 06:48:49  sparker
// Streamlined namespaces
//
// Revision 1.4  2000/04/20 18:56:30  sparker
// Updates to MPM
//
// Revision 1.3  2000/03/30 18:28:52  guilkey
// Moved Material class into Grid directory.  Put indices to velocity
// field and data warehouse into the base class.
//
// Revision 1.1  2000/03/24 00:45:43  guilkey
// Added MPMMaterial class, as well as a skeleton Material class, from
// which MPMMaterial is inherited.  The Material class will be filled in
// as it's mission becomes better identified.
//

