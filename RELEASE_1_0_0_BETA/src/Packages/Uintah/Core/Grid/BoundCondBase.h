#ifndef UINTAH_GRID_BoundCondBase_H
#define UINTAH_GRID_BoundCondBase_H

#include <string>
using std::string;

namespace Uintah {
   
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
    string getType() const { return d_type;};
    
  protected:
    string d_type;
    
  private:
    BoundCondBase(const BoundCondBase&);
    BoundCondBase& operator=(const BoundCondBase&);
  };
} // End namespace Uintah

#endif
