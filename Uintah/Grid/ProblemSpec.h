#ifndef UINTAH_HOMEBREW_ProblemSpec_H
#define UINTAH_HOMEBREW_ProblemSpec_H

#include <Uintah/Grid/Handle.h>
#include <Uintah/Grid/RefCounted.h>

namespace Uintah {
namespace Grid {

class TypeDescription;

// This is the "base" problem spec.  There should be ways of breaking
// this up

/**************************************

CLASS
   ProblemSpec
   
   Short description...

GENERAL INFORMATION

   ProblemSpec.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Problem_Specification

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

class ProblemSpec : public RefCounted {
public:

    const double MAXTIME = .004;

    ProblemSpec();
    virtual ~ProblemSpec();

    double getStartTime() const;
    double getMaximumTime() const;

    static const TypeDescription* getTypeDescription();
private:
    ProblemSpec(const ProblemSpec&);
    ProblemSpec& operator=(const ProblemSpec&);
};

} // end namespace Grid
} // end namespace Uintah

//
// $Log$
// Revision 1.2  2000/03/16 22:08:00  dav
// Added the beginnings of cocoon docs.  Added namespaces.  Did a few other coding standards updates too
//
//

#endif
