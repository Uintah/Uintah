#ifndef UINTAH_HOMEBREW_REFCOUNTED_H
#define UINTAH_HOMEBREW_REFCOUNTED_H

#include <Core/Util/uintahshare.h>
namespace Uintah {
/**************************************

CLASS
   RefCounted
   
   Short description...

GENERAL INFORMATION

   Task.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Reference_Counted, RefCounted, Handle

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

   class UINTAHSHARE RefCounted {
   public:
      RefCounted();
      virtual ~RefCounted();
      
      //////////
      // Insert Documentation Here:
      void addReference() const;
      
      //////////
      // Insert Documentation Here:
      bool removeReference() const;

      int getReferenceCount() const {
	 return d_refCount;
      }

   private:
      RefCounted& operator=(const RefCounted&);
      //////////
      // Insert Documentation Here:
      mutable int d_refCount;
      int d_lockIndex;
   };
} // End namespace Uintah

#endif
