
#ifndef UINTAH_HOMEBREW_ProblemSpecINTERFACE_H
#define UINTAH_HOMEBREW_ProblemSpecINTERFACE_H

#include <Core/Parallel/UintahParallelPort.h>
#include <Core/ProblemSpec/ProblemSpecP.h>

#include <CCA/Ports/uintahshare.h>

#include <string>

namespace Uintah {
/**************************************

CLASS
   ProblemSpecInterface
   
   Short description...

GENERAL INFORMATION

   ProblemSpecInterface.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   ProblemSpec_Interface

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

   class UINTAHSHARE ProblemSpecInterface : public UintahParallelPort {
   public:
      ProblemSpecInterface();
      virtual ~ProblemSpecInterface();
      
      virtual ProblemSpecP readInputFile() = 0;
      virtual std::string getInputFile() = 0;
      
   private:
      ProblemSpecInterface(const ProblemSpecInterface&);
      ProblemSpecInterface& operator=(const ProblemSpecInterface&);
   };
} // End namespace Uintah

#endif

