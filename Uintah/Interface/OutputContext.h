#ifndef UINTAH_HOMEBREW_OutputContext_H
#define UINTAH_HOMEBREW_OutputContext_H

namespace Uintah {
   
   /**************************************
     
     CLASS
       OutputContext
      
       Short Description...
      
     GENERAL INFORMATION
      
       OutputContext.h
      
       Steven G. Parker
       Department of Computer Science
       University of Utah
      
       Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
      
       Copyright (C) 2000 SCI Group
      
     KEYWORDS
       OutputContext
      
     DESCRIPTION
       Long description...
      
     WARNING
      
     ****************************************/
    
   class OutputContext {
   public:
      OutputContext(int fd, long cur)
	 : fd(fd), cur(cur)
      {
      }
      ~OutputContext() {}

      int fd;
      long cur;
   private:
      OutputContext(const OutputContext&);
      OutputContext& operator=(const OutputContext&);
      
   };
} // end namespace Uintah

//
// $Log$
// Revision 1.1  2000/05/15 19:39:53  sparker
// Implemented initial version of DataArchive (output only so far)
// Other misc. cleanups
//
//

#endif

