#ifndef UINTAH_HOMEBREW_OutputContext_H
#define UINTAH_HOMEBREW_OutputContext_H

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#define IRIX
#pragma set woff 1375
#endif
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>

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
      OutputContext(int fd, long cur, ProblemSpecP varnode)
	 : fd(fd), cur(cur), varnode(varnode)
      {
      }
      ~OutputContext() {}

      int fd;
      long cur;
      ProblemSpecP varnode;
   private:
      OutputContext(const OutputContext&);
      OutputContext& operator=(const OutputContext&);
      
   };
} // End namespace Uintah

#endif
