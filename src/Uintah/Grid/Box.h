
#ifndef UINTAH_GRID_BOX_H
#define UINTAH_GRID_BOX_H

#include <SCICore/Geometry/Point.h>
#include <iosfwd>

namespace Uintah {
  namespace Grid {
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

      Box(const Point& lower, const Point& upper);

      bool overlaps(const Box&, double epsilon=1.e-6) const;

      inline Point lower() const {
	return d_lower;
      }
      inline Point upper() const {
	return d_upper;
      }

    private:
      Point d_lower;
      Point d_upper;
    };

  } // end namespace Grid
} // end namespace Uintah

std::ostream& operator<<(std::ostream& out, const Uintah::Grid::Box& b);
    

//
// $Log$
// Revision 1.1  2000/04/13 06:51:01  sparker
// More implementation to get this to work
//
//

#endif
