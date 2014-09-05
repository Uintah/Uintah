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
    BoundCondBase(const string type) : d_type(type) {};
    virtual ~BoundCondBase() {};
    virtual BoundCondBase* clone() = 0;
    string getType() const { return d_type;};
    virtual string getKind() const = 0;
    
  protected:
    string d_type,d_kind;
    
  };
} // End namespace Uintah

#endif
