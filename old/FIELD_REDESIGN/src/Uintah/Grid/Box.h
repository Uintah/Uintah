
#ifndef UINTAH_GRID_BOX_H
#define UINTAH_GRID_BOX_H

#include <SCICore/Geometry/Point.h>
#include <iosfwd>

namespace Uintah {

   using SCICore::Geometry::Point;

/**************************************

  CLASS
        Box
   
  GENERAL INFORMATION

        Box.h

	Steven G. Parker
	Department of Computer Science
	University of Utah

	Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
	
	Copyright (C) 2000 SCI Group

  KEYWORDS
        Box

  DESCRIPTION
        Long description...
  
  WARNING
  
****************************************/

   class Box {
   public:
      Box();
      ~Box();
      Box(const Box&);
      Box& operator=(const Box&);

#if 0
      inline void set( int lx, int ly, int lz,
		       int ux, int uy, int uz ) {
	d_lower.x( lx ); d_lower.y( ly ); d_lower.z( lz );
	d_upper.x( ux ); d_upper.y( uy ); d_upper.z( uz );
      }
#endif
     
      Box(const Point& lower, const Point& upper);

      bool overlaps(const Box&, double epsilon=1.e-6) const;
      bool contains(const Point& p) const {
	 return p.x() >= d_lower.x() && p.y() >= d_lower.y()
	    && p.z() >= d_lower.z() && p.x() < d_upper.x()
	    && p.y() < d_upper.y() && p.z() < d_upper.z();
      }

      inline Point lower() const {
	 return d_lower;
      }
      inline Point upper() const {
	 return d_upper;
      }
      inline Box intersect(const Box& b) const {
	 return Box(SCICore::Geometry::Max(d_lower, b.d_lower),
		    SCICore::Geometry::Min(d_upper, b.d_upper));
      }
      bool degenerate() const {
	 return d_lower.x() >= d_upper.x() || d_lower.y() >= d_upper.y() || d_lower.z() >= d_upper.z();
      }
   private:
      Point d_lower;
      Point d_upper;
   };
   
} // end namespace Uintah

std::ostream& operator<<(std::ostream& out, const Uintah::Box& b);
    

//
// $Log$
// Revision 1.7  2000/06/15 21:57:15  sparker
// Added multi-patch support (bugzilla #107)
// Changed interface to datawarehouse for particle data
// Particles now move from patch to patch
//
// Revision 1.6  2000/05/28 17:25:05  dav
// adding mpi stuff
//
// Revision 1.5  2000/05/10 20:02:59  sparker
// Added support for ghost cells on node variables and particle variables
//  (work for 1 patch but not debugged for multiple)
// Do not schedule fracture tasks if fracture not enabled
// Added fracture directory to MPM sub.mk
// Be more uniform about using IntVector
// Made regions have a single uniform index space - still needs work
//
// Revision 1.4  2000/04/27 23:18:49  sparker
// Added problem initialization for MPM
//
// Revision 1.3  2000/04/26 06:48:47  sparker
// Streamlined namespaces
//
// Revision 1.2  2000/04/25 00:41:21  dav
// more changes to fix compilations
//
// Revision 1.1  2000/04/13 06:51:01  sparker
// More implementation to get this to work
//
//

#endif
