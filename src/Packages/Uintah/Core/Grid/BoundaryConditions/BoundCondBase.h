#ifndef UINTAH_GRID_BoundCondBase_H
#define UINTAH_GRID_BoundCondBase_H

#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>

namespace Uintah {
using std::string;
   
/**************************************

CLASS
   BoundCondBase
   
   
GENERAL INFORMATION

   BoundCondBase.h

   John A. Schmidt
   Department of Mechanical Engineering
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   BoundCondBase

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

  class BoundCondBase  {
  public:
  
    BoundCondBase() {};
    virtual ~BoundCondBase() {};
    virtual BoundCondBase* clone() = 0;
    const string getBCVariable() const { return d_variable; };
    const string getBCType__NEW() const { return d_type__NEW; };
    
  protected:
    string d_variable; // Pressure, Density, etc
    string d_type__NEW; // Dirichlet, Neumann, etc
    
  };
} // End namespace Uintah

#endif
