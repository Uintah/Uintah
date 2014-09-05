#ifndef UINTAH_HOMEBREW_ScatterGatherBase_H
#define UINTAH_HOMEBREW_ScatterGatherBase_H

namespace Uintah {
   /**************************************
     
     CLASS
       ScatterGatherBase
      
       Short Description...
      
     GENERAL INFORMATION
      
       ScatterGatherBase.h
      
       Steven G. Parker
       Department of Computer Science
       University of Utah
      
       Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
      
       Copyright (C) 2000 SCI Group
      
     KEYWORDS
       ScatterGatherBase
      
     DESCRIPTION
       Long description...
      
     WARNING
      
     ****************************************/
    
   class ScatterGatherBase {
   public:
      ScatterGatherBase();
      virtual ~ScatterGatherBase();
   private:
      ScatterGatherBase(const ScatterGatherBase&);
      ScatterGatherBase& operator=(const ScatterGatherBase&);
      
   };

} // End namespace Uintah

#endif
