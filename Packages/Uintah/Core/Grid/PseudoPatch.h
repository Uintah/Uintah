#ifndef UINTAH_HOMEBREW_PseudoPatch_H
#define UINTAH_HOMEBREW_PseudoPatch_H

#include <iostream>
using namespace std;

#include <Core/Geometry/IntVector.h>
#include <Packages/Uintah/Core/Grid/share.h>

namespace Uintah {

  using namespace SCIRun;

   
/**************************************
      
CLASS
   PseudoPatch
      
   A light weight patch 

GENERAL INFORMATION
      
   PseudoPatch.h
      
   Justin Luitjens
   Department of Computer Science
   University of Utah
      
   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
      
   Copyright (C) 2000 SCI Group
      
KEYWORDS
   Patch
      
DESCRIPTION
   Long description...
      
WARNING
     
****************************************/
    
  class SCISHARE PseudoPatch {
    public:
   
    inline int getArea(int side) const
    {
      int area = 1;
      for (int i = 0; i < 3; i++)
        if (i != side)
          area *= high[i] - low[i];
      return area;
    }
    
    inline int getVolume() const
    {
      return (high.x() - low.x()) * (high.y() - low.y()) * (high.z() - low.z());
    }
     
    // Locations in space of opposite box corners.
    IntVector low;
    IntVector high;

   }; // end class PseudoPatch

   inline bool operator<(const PseudoPatch& p1, const PseudoPatch& p2) 
   {
     return p1.getVolume()<p2.getVolume();
   }

   inline std::ostream& operator<<(std::ostream& out, const Uintah::PseudoPatch& p1) 
   {
     out << "{" << p1.low << " " << p1.high << " (" << p1.getVolume() << ")}";
     return out;
   }

} // End namespace Uintah

#endif
