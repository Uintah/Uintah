#ifndef Uintah_Parallel_UintahParallelComponent_h
#define Uintah_Parallel_UintahParallelComponent_h

#include <string>
using std::string;

namespace Uintah {
namespace Parallel {

class UintahParallelPort;

/**************************************

CLASS
   UintahParallelComponent
   
   Short description...

GENERAL INFORMATION

   UintahParallelComponent.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Uintah_Parallel_Component, Component

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

class UintahParallelComponent {
public:
    UintahParallelComponent();
    virtual ~UintahParallelComponent();

    //////////
    // Insert Documentation Here:
    void setPort(const string& name, UintahParallelPort* port);
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
