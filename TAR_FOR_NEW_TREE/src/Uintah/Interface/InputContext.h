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
} // end namespace Uintah

//
// $Log$
// Revision 1.1  2000/05/20 08:09:36  sparker
// Improved TypeDescription
// Finished I/O
// Use new XML utility libraries
//
//

#endif

