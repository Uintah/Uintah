#ifndef UINTAH_HOMEBREW_OUTPUT_H
#define UINTAH_HOMEBREW_OUTPUT_H

#include <Uintah/Parallel/UintahParallelPort.h>
#include <Uintah/Interface/DataWarehouseP.h>
#include <Uintah/Grid/LevelP.h>
#include <Uintah/Interface/SchedulerP.h>

namespace Uintah {
namespace Interface {

using Uintah::Parallel::UintahParallelPort;
using Uintah::Grid::LevelP;

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
// Revision 1.5  2000/04/11 07:10:53  sparker
// Completing initialization and problem setup
// Finishing Exception modifications
//
// Revision 1.4  2000/03/17 09:30:03  sparker
// New makefile scheme: sub.mk instead of Makefile.in
// Use XML-based files for module repository
// Plus many other changes to make these two things work
//
// Revision 1.3  2000/03/16 22:08:23  dav
// Added the beginnings of cocoon docs.  Added namespaces.  Did a few other coding standards updates too
//
//

#endif
