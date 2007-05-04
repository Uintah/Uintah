
#ifndef Uintah_TableFactory_h
#define Uintah_TableFactory_h

#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>

namespace Uintah {
  class TableInterface;

/****************************************

CLASS
   TableFactory
   
   Short description...

GENERAL INFORMATION

   TableFactory.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   TableFactory

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

  class TableFactory {
  public:

    static TableInterface* readTable(const ProblemSpecP& params,
				     const string& name);
  private:
    TableFactory();
  };
} // End namespace Uintah
    
#endif
