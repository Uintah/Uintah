#ifndef UINTAH_PARALLEL_UINTAHPARALLELPORT_H
#define UINTAH_PARALLEL_UINTAHPARALLELPORT_H

#include <Packages/Uintah/Core/ProblemSpec/RefCounted.h>

namespace Uintah {

/**************************************

CLASS
   ParallelPort
   
   Short description...

GENERAL INFORMATION

   ParallelPort.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   _Parallel_Port

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

class UintahParallelPort : public RefCounted {
public:
           UintahParallelPort();
  virtual ~UintahParallelPort();
};

} // End namespace Uintah

#endif
