#ifndef UINTAH_GRID_NeighBoundCond_H
#define UINTAH_GRID_NeighBoundCond_H

#include <Core/Grid/BoundaryConditions/BoundCondBase.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Malloc/Allocator.h>

namespace Uintah {
   
/**************************************

CLASS
   NeighBoundCond
   
   
GENERAL INFORMATION

   NeighBoundCond.h

   John A. Schmidt
   Department of Mechanical Engineering
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   NeighBoundCond

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

  class NeighBoundCond : public BoundCondBase {
  public:
    NeighBoundCond() : BoundCondBase("Neighbor") {}; 
    NeighBoundCond(ProblemSpecP&) {};
    virtual ~NeighBoundCond() {};
    virtual NeighBoundCond* clone() {return scinew NeighBoundCond(*this);};
    virtual string getKind() const { return "";};
      
  private:
#if 0
    NeighBoundCond(const NeighBoundCond&);
    NeighBoundCond& operator=(const NeighBoundCond&);
#endif
      
     
   };
} // End namespace Uintah

#endif
