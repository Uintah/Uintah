#ifndef UINTAH_HOMEBREW_InputContext_H
#define UINTAH_HOMEBREW_InputContext_H

namespace Uintah {
   /**************************************
     
     CLASS
       InputContext
      
       Short Description...
      
     GENERAL INFORMATION
      
       InputContext.h
      
       Steven G. Parker
       Department of Computer Science
       University of Utah
      
       Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
      
       Copyright (C) 2000 SCI Group
      
     KEYWORDS
       InputContext
      
     DESCRIPTION
       Long description...
      
     WARNING
      
     ****************************************/
    
   class InputContext {
   public:
      InputContext(int fd, long cur)
	 : fd(fd), cur(cur)
      {
      }
      ~InputContext() {}

      int fd;
      long cur;
   private:
      InputContext(const InputContext&);
      InputContext& operator=(const InputContext&);
      
   };
} // End namespace Uintah

#endif
