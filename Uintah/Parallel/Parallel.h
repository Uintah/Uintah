#ifndef UINTAH_HOMEBREW_PARALLEL_H
#define UINTAH_HOMEBREW_PARALLEL_H

namespace Uintah {
namespace Parallel {

/**************************************

CLASS
   Parallel
   
   Short description...

GENERAL INFORMATION

   Parallel.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Parallel

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

class Parallel {
public:
    //////////
    // Insert Documentation Here:
    static void initializeManager(int argc, char** argv);

    //////////
    // Insert Documentation Here:
    static void finalizeManager();

private:
    Parallel();
    Parallel(const Parallel&);
    ~Parallel();
    Parallel& operator=(const Parallel&);
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
