#ifndef Uintah_PropertyBase_h
#define Uintah_PropertyBase_h

namespace Uintah {
  class ICEMaterial;
/**************************************

CLASS
   PropertyBase
   
   Short description...

GENERAL INFORMATION

   PropertyBase.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS


DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

  class PropertyBase {
  public:
    virtual ~PropertyBase();
    void setICEMaterial(ICEMaterial* matl) {
      ice_matl = matl;
    }

  protected:
    PropertyBase();
    ICEMaterial* ice_matl;
  };
} // End namespace Uintah
      
#endif  // Uintah_PropertyBase_h


