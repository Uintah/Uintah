#ifndef UINTAH_HOMEBREW_Region_H
#define UINTAH_HOMEBREW_Region_H

#include "Array3Index.h"

#include <Uintah/Grid/SubRegion.h>
#include <Uintah/Grid/ParticleSet.h>

#include <SCICore/Geometry/Point.h>
#include <SCICore/Geometry/Vector.h>


#include <iostream> // TEMPORARY

using SCICore::Geometry::Point;
using SCICore::Geometry::Vector;

namespace Uintah {
namespace Grid {

class NodeIterator;
class NodeSubIterator;

/**************************************

CLASS
   Region
   
   Short Description...

GENERAL INFORMATION

   Region.h

   Steven G. Parker
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

class Region {
public:

    //////////
    // Insert Documentation Here:
    Vector dCell() const;

    //////////
    // Insert Documentation Here:
    void findCell(const Vector& pos, int& ix, int& iy, int& iz) const;

    //////////
    // Insert Documentation Here:
    bool findCellAndWeights(const SCICore::Geometry::Vector& pos,
			    Array3Index ni[8], double S[8]) const;
    //////////
    // Insert Documentation Here:
    bool findCellAndShapeDerivatives(const SCICore::Geometry::Vector& pos,
				     Array3Index ni[8],
				     SCICore::Geometry::Vector S[8]) const;
    //////////
    // Insert Documentation Here:
    inline NodeIterator begin() const;

    //////////
    // Insert Documentation Here:
    inline NodeIterator end() const;

    //////////
    // Insert Documentation Here:
    void subregionIteratorPair(int i, int n,
			       NodeSubIterator& iter,
			       NodeSubIterator& end) const;
    //////////
    // Insert Documentation Here:
    SubRegion subregion(int i, int n) const;

    //////////
    // Insert Documentation Here:
    inline int getNx() const {
	return d_nx;
    }

    //////////
    // Insert Documentation Here:
    inline int getNy() const {
	return d_ny;
    }

    //////////
    // Insert Documentation Here:
    inline int getNz() const {
	return d_nz;
    }

    //////////
    // Insert Documentation Here:
    inline bool contains(const Array3Index& idx) const {
	return idx.i() >= 0 && idx.j() >= 0 && idx.k() >= 0
	    && idx.i() <= d_nx && idx.j() <= d_ny && idx.k() <= d_nz;
    }
protected:
    friend class Level;

    //////////
    // Insert Documentation Here:
    Region(const SCICore::Geometry::Point& min,
	   const SCICore::Geometry::Point& max,
	   int nx, int ny, int nz);
    ~Region();

private:
    Region(const Region&);
    Region& operator=(const Region&);

    //////////
    // Insert Documentation Here:
    Point d_lower;
    Point d_upper;

    //////////
    // Insert Documentation Here:
    int d_nx, d_ny, d_nz;

    friend class NodeIterator;
};

} // end namespace Grid
} // end namespace Uintah

#include "NodeIterator.h"

namespace Uintah {
namespace Grid {

inline NodeIterator Region::begin() const
{
    return NodeIterator(this, 0, 0, 0);
}

inline NodeIterator Region::end() const
{
    return NodeIterator(this, d_nx+1, d_ny+1, d_nz+1);
}

} // end namespace Grid
} // end namespace Uintah

//
// $Log$
// Revision 1.3  2000/03/21 01:29:42  dav
// working to make MPM stuff compile successfully
//
// Revision 1.2  2000/03/16 22:08:01  dav
// Added the beginnings of cocoon docs.  Added namespaces.  Did a few other coding standards updates too
//
//

#endif
