#ifndef UINTAH_HOMEBREW_OutputContext_H
#define UINTAH_HOMEBREW_OutputContext_H

#ifdef __sgi
#define IRIX
#pragma set woff 1375
#endif
#include <dom/DOM_Element.hpp>

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
      OutputContext(int fd, long cur, const DOM_Element& varnode)
	 : fd(fd), cur(cur), varnode(varnode)
      {
      }
      ~OutputContext() {}

      int fd;
      long cur;
      DOM_Element varnode;
   private:
      OutputContext(const OutputContext&);
      OutputContext& operator=(const OutputContext&);
      
   };
} // end namespace Uintah

//
// $Log$
// Revision 1.2  2000/05/20 08:09:36  sparker
// Improved TypeDescription
// Finished I/O
// Use new XML utility libraries
//
// Revision 1.1  2000/05/15 19:39:53  sparker
// Implemented initial version of DataArchive (output only so far)
// Other misc. cleanups
//
//

#endif

