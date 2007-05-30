#ifndef UINTAH_HOMEBREW_Region_H
#define UINTAH_HOMEBREW_Region_H

#include <SCIRun/Core/Geometry/IntVector.h>

#undef None

#include <sgi_stl_warnings_off.h>
#include   <string>
#include   <deque>
#include   <map>
#include   <iosfwd>
#include   <iostream>
#include   <vector>
#include <sgi_stl_warnings_on.h>

#include <Core/Grid/share.h>

namespace Uintah {

  using SCIRun::IntVector;
  using namespace std;

/**************************************
      
CLASS
   Region
      
   An integer region of space formed by 2 corners
   
GENERAL INFORMATION
      
   Region.h
      
   Justin Luitjens
   Department of Computer Science
   University of Utah
      
   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
      
   Copyright (C) 2000 SCI Group
      
KEYWORDS
   Region

DESCRIPTION
   Long description...
      
WARNING
     
****************************************/
    
  class SCISHARE Region {
    public:
    
   inline Region()
    : d_lowIndex(IntVector(0,0,0)), d_highIndex(IntVector(0,0,0))
    {
    }
    
    inline Region(const Region& copy)
    : d_lowIndex(copy.d_lowIndex), d_highIndex(copy.d_highIndex)
    {
    }

    inline Region& operator=(const Region& copy)
    {
      d_lowIndex = copy.d_lowIndex;
      d_highIndex = copy.d_highIndex;
      return *this;
    }
    
    inline Region(const IntVector &low, const IntVector &high)
    : d_lowIndex(low), d_highIndex(high)
    {
    }
    
    //Get refrences to the indices
    inline IntVector& low()
    { return d_lowIndex;  }
    
    inline IntVector& high()
    { return d_highIndex;  }

    /********************
      The following are needed in order to use Patch as a Box in
      Core/Container/SuperBox.h (see
      Packages/Uintah/Core/Grid/Variables/LocallyComputedPatchVarMap.cc)
    *********************/
    
    inline IntVector getLow() const
    { return d_lowIndex; }
    
    inline IntVector getHigh() const
    { return d_highIndex; }

    inline int getVolume() const
    { return getVolume(getLow(), getHigh()); }
    
    inline int getArea(int side) const
    {
      int area = 1;
      for (int i = 0; i < 3; i++)
        if (i != side)
          area *= getHigh()[i] - getLow()[i];
      return area;
    }
    
    static inline int getVolume(const IntVector& low, const IntVector& high)
    {
      return (high.x() -  low.x()) * (high.y() - low.y()) * (high.z() - low.z());
    }

    /*****************
      Stolen from Box.h
    ******************/

    bool overlaps(const Region&) const;
    inline Region intersect(const Region& b) const {
      return Region(Max(d_lowIndex, b.d_lowIndex),
               Min(d_highIndex, b.d_highIndex));
    }
 
    inline bool degenerate() const {
      return d_lowIndex.x() >= d_highIndex.x() || d_lowIndex.y() >= d_highIndex.y() || d_lowIndex.z() >= d_highIndex.z();
    }

    static deque<Region> difference(const Region& b1, const Region& b2);
    static deque<Region> difference(deque<Region>& region1, deque<Region>& region22);
   
    class VolumeCompare {
      public:
        inline bool operator()(const Region& p1, const Region& p2) const {
          return p1.getVolume() < p2.getVolume();
        }
    };

    protected:
     //Note:  Do not add variables to this class because it will slow down MPI transfers

     //////////
     // Locations in space of opposite corners.
     IntVector d_lowIndex;
     IntVector d_highIndex;
     
   }; // end class Region

  
   inline std::ostream& operator<<(std::ostream& out, const Uintah::Region& p1)
   {
     out << "{" << p1.getLow() << " " << p1.getHigh() << " (" << p1.getVolume() << ")}";
     return out;
   }

} // End namespace Uintah

#endif
