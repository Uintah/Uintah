#ifndef UINTAH_HOMEBREW_SubPatch_H
#define UINTAH_HOMEBREW_SubPatch_H

#include <Packages/Uintah/Core/Grid/Array3Index.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>

namespace Uintah {

using namespace SCIRun;

/**************************************

CLASS
   SubPatch
   
   I THINK THIS IS CORRECT:
   The SubPatch Class is only a specification of a sub-patch's
   geometrical location (specified by an "upper" and "lower" point
   location in the real world) and the sub-patches storage location
   (index) into a larger 3D array.

GENERAL INFORMATION

   SubPatch.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   SubPatch

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

   class SubPatch {
   public:
      SubPatch(const Point& lower,
	       const Point& upper,
	       int sx, int sy, int sz,
	       int ex, int ey, int ez);
      ~SubPatch();
      SubPatch(const SubPatch&);
      SubPatch& operator=(const SubPatch&);
      
      //////////
      // Determines if the Point "p" is geometrically within this SubPatch
      inline bool contains(const Point& p) const {
	 return p.x() >= d_lower.x() && 
	    p.y() >= d_lower.y() &&
	    p.z() >= d_lower.z() &&
	    p.x() < d_upper.x()  &&
	    p.y() < d_upper.y()  &&
	    p.z() < d_upper.z();
      }
      ////////// 
      // Determines if the Array index "idx" is "part of" this SubPatch
      inline bool contains(const Array3Index& idx) const {
	 return idx.i() >= d_sx && idx.j() >= d_sy && idx.k() >= d_sz
	    && idx.i() <= d_ex && idx.j() <= d_ey && idx.k() <= d_ez;
      }
   private:
      Point d_lower;
      Point d_upper;
      
      ////////// 
      // Starting and ending indexes into a larger 3D array that stores
      // the data for this sub-patch.
      int d_sx, d_sy, d_sz;
      int d_ex, d_ey, d_ez;
   };

} // End namespace Uintah

#endif
