#ifndef UINTAH_HOMEBREW_OUTPUT_H
#define UINTAH_HOMEBREW_OUTPUT_H

#include <Uintah/Parallel/UintahParallelPort.h>
#include <Uintah/Interface/OutputP.h>
#include <Uintah/Interface/DataWarehouseP.h>
#include <Uintah/Grid/LevelP.h>
#include <Uintah/Interface/SchedulerP.h>

namespace Uintah {
namespace Interface {

using Uintah::Parallel::UintahParallelPort;

/**************************************

CLASS
   Output
   
   Short description...

GENERAL INFORMATION

   Output.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Output

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

class Output : public UintahParallelPort {
public:
    Output();
    virtual ~Output();

    //////////
    // Insert Documentation Here:
    void finalizeTimestep(double t, double delt, const LevelP&, SchedulerP&,
			  const DataWarehouseP&);
private:
    Output(const Output&);
    Output& operator=(const Output&);
};

} // end namespace Interface
} // end namespace Uintah

//
// $Log$
// Revision 1.3  2000/03/16 22:08:23  dav
// Added the beginnings of cocoon docs.  Added namespaces.  Did a few other coding standards updates too
//
//

#endif
