#ifndef UINTAH_PARALLEL_UINTAHPARALLELPORT_H
#define UINTAH_PARALLEL_UINTAHPARALLELPORT_H

#include <Packages/Uintah/Grid/RefCounted.h>

namespace Uintah {
/**************************************

CLASS
   Packages/UintahParallelPort
   
   Short description...

GENERAL INFORMATION

   Packages/UintahParallelPort.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Packages/Uintah_Parallel_Port

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

   class Packages/UintahParallelPort : public RefCounted {
   public:
      Packages/UintahParallelPort();
      virtual ~Packages/UintahParallelPort();
   };
} // End namespace Uintah



#endif
