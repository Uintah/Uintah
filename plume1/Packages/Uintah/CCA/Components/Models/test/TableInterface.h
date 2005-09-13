
#ifndef Uintah_TableInterface_h
#define Uintah_TableInterface_h

#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>

namespace Uintah {

/****************************************

CLASS
   TableInterface
   
   Short description...

GENERAL INFORMATION

   TableInterface.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   TableInterface

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

  class TableInterface {
  public:
    TableInterface();
    virtual ~TableInterface();

    virtual void addIndependentVariable(const string&) = 0;
    virtual int addDependentVariable(const string&) = 0;
    
    virtual void setup() = 0;
    
    virtual void interpolate(int index, CCVariable<double>& result,
			     const CellIterator&,
			     vector<constCCVariable<double> >& independents) = 0;
    virtual double interpolate(int index, vector<double>& independents) = 0;
  private:
  };
} // End namespace Uintah
    
#endif
