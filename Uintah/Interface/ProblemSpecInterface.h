
#ifndef UINTAH_HOMEBREW_ProblemSpecINTERFACE_H
#define UINTAH_HOMEBREW_ProblemSpecINTERFACE_H

#include <Uintah/Parallel/UintahParallelPort.h>
#include <Uintah/Interface/ProblemSpecP.h>

namespace Uintah {
namespace Interface {

using Uintah::Parallel::UintahParallelPort;

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

class ProblemSpecInterface : public UintahParallelPort {
public:
    ProblemSpecInterface();
    virtual ~ProblemSpecInterface();

    virtual ProblemSpecP readInputFile() = 0;

private:
    ProblemSpecInterface(const ProblemSpecInterface&);
    ProblemSpecInterface& operator=(const ProblemSpecInterface&);
};

} // end namespace Interface
} // end namespace Uintah

//
// $Log$
// Revision 1.1  2000/04/11 07:10:54  sparker
// Completing initialization and problem setup
// Finishing Exception modifications
//
//

#endif

