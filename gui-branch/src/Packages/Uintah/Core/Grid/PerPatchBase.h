
#ifndef UINTAH_HOMEBREW_PerPatchBase_H
#define UINTAH_HOMEBREW_PerPatchBase_H

#include <string>

namespace Uintah {
  using namespace std;
  class Patch;
  class RefCounted;

/**************************************

CLASS
   PerPatchBase
   
   Short description...

GENERAL INFORMATION

   PerPatchBase.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   PerPatchBase

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

   class PerPatchBase {
   public:
      
      virtual ~PerPatchBase();
      
      virtual void copyPointer(const PerPatchBase&) = 0;
      virtual PerPatchBase* clone() const = 0;
      virtual RefCounted* getRefCounted();
      virtual void getSizeInfo(string& elems, unsigned long& totsize,
			       void*& ptr) const = 0;
   protected:
      PerPatchBase(const PerPatchBase&);
      PerPatchBase();
      
   private:
      PerPatchBase& operator=(const PerPatchBase&);
   };
} // End namespace Uintah

#endif
