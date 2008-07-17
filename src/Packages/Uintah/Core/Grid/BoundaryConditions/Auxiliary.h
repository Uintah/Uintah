#ifndef UINTAH_GRID_AuxiliaryBoundCond_H
#define UINTAH_GRID_AuxiliaryBoundCond_H

#include <Packages/Uintah/Core/Grid/BoundaryConditions/BoundCondBase.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Malloc/Allocator.h>

namespace Uintah {
   
/**************************************

CLASS
   AuxiliaryBoundCond
   
   
GENERAL INFORMATION

   AuxiliaryBoundCond.h

   John A. Schmidt
   Department of Mechanical Engineering
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   AuxiliaryBoundCond

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

  class AuxiliaryBoundCond : public BoundCondBase  {
  public:
    AuxiliaryBoundCond():BoundCondBase("Auxiliary") {};
    AuxiliaryBoundCond(ProblemSpecP&) {d_type = "Auxiliary";};
    virtual ~AuxiliaryBoundCond() {};
    virtual AuxiliaryBoundCond* clone() {return scinew AuxiliaryBoundCond(*this);};
    virtual string getKind() const {return "auxiliary";};
  private:
#if 0
    AuxiliaryBoundCond(const AuxiliaryBoundCond&);
    AuxiliaryBoundCond& operator=(const AuxiliaryBoundCond&);
#endif
     
   };
} // End namespace Uintah

#endif
