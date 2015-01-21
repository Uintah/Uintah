#ifndef UINTAH_GRID_SymmetryBoundCond_H
#define UINTAH_GRID_SymmetryBoundCond_H

#include <Core/Grid/BoundaryConditions/BoundCondBase.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Malloc/Allocator.h>

namespace Uintah {
   
/**************************************

CLASS
   SymmetryBoundCond
   
   
GENERAL INFORMATION

   SymmetryBoundCond.h

   John A. Schmidt
   Department of Mechanical Engineering
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   SymmetryBoundCond

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

  class SymmetryBoundCond : public BoundCondBase  {
  public:
    SymmetryBoundCond():BoundCondBase("Symmetry") {};
    SymmetryBoundCond(ProblemSpecP&) {d_type = "Symmetric";};
    virtual ~SymmetryBoundCond() {};
    virtual SymmetryBoundCond* clone() {return scinew SymmetryBoundCond(*this);};
    virtual string getKind() const {return "symmetry";};
  private:
#if 0
    SymmetryBoundCond(const SymmetryBoundCond&);
    SymmetryBoundCond& operator=(const SymmetryBoundCond&);
#endif
     
   };
} // End namespace Uintah

#endif
