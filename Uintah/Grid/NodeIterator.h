
#ifndef UINTAH_HOMEBREW_NodeIterator_H
#define UINTAH_HOMEBREW_NodeIterator_H

#include "Array3Index.h"
#include "Region.h"

class NodeIterator {
public:
    inline ~NodeIterator() {
    }
    inline NodeIterator operator++(int) {
	NodeIterator old(*this);
	if(++iz > region->nz){
	    iz=0;
	    if(++iy > region->ny){
		iy=0;
		if(++ix > region->nx){
		    iy=region->ny+1; iz=region->nz+1;
		}
	    }
	}
	return old;
    }
    inline Array3Index operator*() const {
	return Array3Index(ix, iy, iz);
    }
    inline bool operator!=(const NodeIterator& n) const {
	return ix != n.ix || iy != n.iy || iz != n.iz;
    }
private:
    NodeIterator();
    inline NodeIterator(const NodeIterator& copy)
	: region(copy.region), ix(copy.ix), iy(copy.iy), iz(copy.iz) {
    }
    NodeIterator& operator=(const NodeIterator& copy);
    inline NodeIterator(const Region* region, int ix, int iy, int iz)
	: region(region), ix(ix), iy(iy), iz(iz) {
    }
	
    friend class Region;
    const Region* region;
    int ix, iy, iz;
};

#endif
