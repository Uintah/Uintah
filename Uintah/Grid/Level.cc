
#include "Level.h"
#include "Handle.h"
#include "Region.h"
#include <iostream>
using SCICore::Geometry::Point;
using std::cerr;

Level::Level()
{
}

Level::~Level()
{
    // Delete all of the regions managed by this level
    for(regionIterator iter=regions.begin(); iter != regions.end(); iter++)
	delete *iter;
}

Level::const_regionIterator Level::regionsBegin() const
{
    return regions.begin();
}

Level::const_regionIterator Level::regionsEnd() const
{
    return regions.end();
}

Level::regionIterator Level::regionsBegin()
{
    return regions.begin();
}

Level::regionIterator Level::regionsEnd()
{
    return regions.end();
}

Region* Level::addRegion(const Point& min, const Point& max,
			 int nx, int ny, int nz)
{
    Region* r=new Region(min, max, nx, ny, nz);
    regions.push_back(r);
    return r;
}
