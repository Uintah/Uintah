#ifndef UINTAH_GRID_BCDataBase_H
#define UINTAH_GRID_BCDataBase_H

#include <SCIRun/Core/Geometry/IntVector.h>
#include <SCIRun/Core/Geometry/Point.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

namespace Uintah {

  using SCIRun::IntVector;
  using SCIRun::Point;
   
/**************************************

CLASS
   BCDataBase
   
   
GENERAL INFORMATION

   BCDataBase.h

   John A. Schmidt
   Department of Mechanical Engineering
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   BCDataBase

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

  class BCDataBase {
  public:
    BCDataBase() {};
    virtual ~BCDataBase() {};    
    virtual BCDataBase* clone() = 0;
    virtual void getBCData(BCData& bc) const = 0;
    virtual void addBCData(BCData& bc)  = 0;
    virtual void setBoundaryIterator(std::vector<IntVector>& b) = 0;
    virtual void setInteriorIterator(std::vector<IntVector>& i) = 0;
    virtual void getBoundaryIterator(std::vector<IntVector>& b) const = 0;
    virtual void getInteriorIterator(std::vector<IntVector>& i) const = 0;
    virtual bool inside(const Point& p) const = 0;
  };
} // End namespace Uintah

#endif
