
#ifndef UINTAH_HOMEBREW_Region_H
#define UINTAH_HOMEBREW_Region_H

#include <SCICore/Geometry/Point.h>
#include <SCICore/Geometry/Vector.h>
#include "Array3Index.h"
#include "ParticleSet.h"
#include <iostream> // TEMPORARY
class NodeIterator;
class NodeSubIterator;
class Array3Index;
class SubRegion;

class Region {
public:
    SCICore::Geometry::Vector dCell() const;
    void findCell(const SCICore::Geometry::Vector& pos,
		  int& ix, int& iy, int& iz) const;
    bool findCellAndWeights(const SCICore::Geometry::Vector& pos,
			    Array3Index ni[8], double S[8]) const;
    bool findCellAndShapeDerivatives(const SCICore::Geometry::Vector& pos,
				     Array3Index ni[8],
				     SCICore::Geometry::Vector S[8]) const;
    inline NodeIterator begin() const;
    inline NodeIterator end() const;
    void subregionIteratorPair(int i, int n,
			       NodeSubIterator& iter,
			       NodeSubIterator& end) const;
    SubRegion subregion(int i, int n) const;
    inline int getNx() const {
	return nx;
    }
    inline int getNy() const {
	return ny;
    }
    inline int getNz() const {
	return nz;
    }
    inline bool contains(const Array3Index& idx) const {
	return idx.i() >= 0 && idx.j() >= 0 && idx.k() >= 0
	    && idx.i() <= nx && idx.j() <= ny && idx.k() <= nz;
    }
protected:
    friend class Level;
    Region(const SCICore::Geometry::Point& min, const SCICore::Geometry::Point& max,
	   int nx, int ny, int nz);
    ~Region();

private:
    Region(const Region&);
    Region& operator=(const Region&);

    SCICore::Geometry::Point lower;
    SCICore::Geometry::Point upper;
    int nx, ny, nz;

    friend class NodeIterator;
};

#include "NodeIterator.h"

inline NodeIterator Region::begin() const
{
    return NodeIterator(this, 0, 0, 0);
}

inline NodeIterator Region::end() const
{
    return NodeIterator(this, nx+1, ny+1, nz+1);
}

#endif
