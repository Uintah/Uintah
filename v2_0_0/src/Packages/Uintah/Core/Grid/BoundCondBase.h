#ifndef UINTAH_GRID_BoundCondBase_H
#define UINTAH_GRID_BoundCondBase_H

#include <Packages/Uintah/Core/ProblemSpec/RefCounted.h>
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

  class BoundCondBase : public RefCounted  {
  public:
    BoundCondBase() {};
    BoundCondBase(const string type) : d_type(type) {};
    virtual ~BoundCondBase() {};
    string getType() const { return d_type;};
    
  protected:
    string d_type;
    
  private:
#if 0
    BoundCondBase(const BoundCondBase&);
    BoundCondBase& operator=(const BoundCondBase&);
#endif
  };
} // End namespace Uintah

#endif
