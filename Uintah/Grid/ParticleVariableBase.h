
#ifndef UINTAH_HOMEBREW_ParticleVariableBase_H
#define UINTAH_HOMEBREW_ParticleVariableBase_H


namespace Uintah {

   class Region;

/**************************************

CLASS
   ParticleVariableBase
   
   Short description...

GENERAL INFORMATION

   ParticleVariableBase.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   ParticleVariableBase

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

   class ParticleVariableBase {
   public:
      
      virtual ~ParticleVariableBase();
      virtual void copyPointer(const ParticleVariableBase&) = 0;
      
      //////////
      // Insert Documentation Here:
      virtual ParticleVariableBase* clone() const = 0;
      
   protected:
      ParticleVariableBase(const ParticleVariableBase&);
      ParticleVariableBase();
      
   private:
      ParticleVariableBase& operator=(const ParticleVariableBase&);
   };
   
} // end namespace Uintah

//
// $Log$
// Revision 1.3  2000/04/28 07:35:37  sparker
// Started implementation of DataWarehouse
// MPM particle initialization now works
//
// Revision 1.2  2000/04/26 06:48:52  sparker
// Streamlined namespaces
//
// Revision 1.1  2000/04/20 20:09:22  jas
// I don't know what these do, but Steve says we need them.
//
// Revision 1.1  2000/04/19 05:26:15  sparker
// Implemented new problemSetup/initialization phases
// Simplified DataWarehouse interface (not finished yet)
// Made MPM get through problemSetup, but still not finished
//
//

#endif
