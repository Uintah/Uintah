#ifndef UINTAH_HOMEBREW_SubPatch_H
#define UINTAH_HOMEBREW_SubPatch_H

#include "Array3Index.h"
#include <SCICore/Geometry/Point.h>
#include <SCICore/Geometry/Vector.h>

namespace Uintah {

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
      SubPatch(const SCICore::Geometry::Point& lower,
		const SCICore::Geometry::Point& upper,
		int sx, int sy, int sz,
		int ex, int ey, int ez);
      ~SubPatch();
      SubPatch(const SubPatch&);
      SubPatch& operator=(const SubPatch&);
      
      //////////
      // Determines if the Point "p" is geometrically within this SubPatch
      inline bool contains(const SCICore::Geometry::Point& p) const {
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
      SCICore::Geometry::Point d_lower;
      SCICore::Geometry::Point d_upper;
      
      ////////// 
      // Starting and ending indexes into a larger 3D array that stores
      // the data for this sub-patch.
      int d_sx, d_sy, d_sz;
      int d_ex, d_ey, d_ez;
   };
   
} // end namespace Uintah

//
// $Log$
// Revision 1.1  2000/05/30 20:19:34  sparker
// Changed new to scinew to help track down memory leaks
// Changed region to patch
//
// Revision 1.4  2000/04/26 06:48:59  sparker
// Streamlined namespaces
//
// Revision 1.3  2000/03/17 18:45:42  dav
// fixed a few more namespace problems
//
// Revision 1.2  2000/03/16 22:08:01  dav
// Added the beginnings of cocoon docs.  Added namespaces.  Did a few other coding standards updates too
//
//

#endif
