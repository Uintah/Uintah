#ifndef UINTAH_PARALLEL_UINTAHPARALLELPORT_H
#define UINTAH_PARALLEL_UINTAHPARALLELPORT_H

#include <Uintah/Grid/RefCounted.h>

namespace Uintah {
namespace Parallel {

using Uintah::Grid::RefCounted;

/**************************************

CLASS
   UintahParallelPort
   
   Short description...

GENERAL INFORMATION

   UintahParallelPort.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Uintah_Parallel_Port

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

class UintahParallelPort : public RefCounted {
public:
    UintahParallelPort();
    virtual ~UintahParallelPort();
};

} // end namespace Parallel
} // end namespace Uintah

//
// $Log$
// Revision 1.2  2000/03/16 22:08:39  dav
// Added the beginnings of cocoon docs.  Added namespaces.  Did a few other coding standards updates too
//
//

#endif
