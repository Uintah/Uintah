
#ifndef UINTAH_HOMEBREW_NodeSubIterator_H
#define UINTAH_HOMEBREW_NodeSubIterator_H

#include "Array3Index.h"

class NodeSubIterator {
public:
    inline NodeSubIterator() {
    }
    inline ~NodeSubIterator() {
    }
    inline NodeSubIterator operator++(int) {
	NodeSubIterator old(*this);
	if(++iz >= ez){
	    iz=sz;
	    if(++iy >= ey){
		iy=sy;
		if(++ix >= ex){
		    iy=ey; iz=ez;
		}
	    }
	}
	return old;
    }
    inline Array3Index operator*() const {
	return Array3Index(ix, iy, iz);
    }
    inline bool operator!=(const NodeSubIterator& n) const {
	return ix != n.ix || iy != n.iy || iz != n.iz;
    }
    inline int x() const {
	return ix;
    }
    inline int y() const {
	return iy;
    }
    inline int z() const {
	return iz;
    }
private:
    inline NodeSubIterator(const NodeSubIterator& copy)
	: ix(copy.ix), iy(copy.iy), iz(copy.iz),
	  sx(copy.sx), sy(copy.sy), sz(copy.sz),
	  ex(copy.ex), ey(copy.ey), ez(copy.ez)
    {
    }
    inline NodeSubIterator& operator=(const NodeSubIterator& copy) {
	ix = copy.ix; iy = copy.iy; iz = copy.iz;
	sx = copy.sx; sy = copy.sy; sz = copy.sz;
	ex = copy.ex; ey = copy.ey; ez = copy.ez;
	return *this;
    }
    inline NodeSubIterator(int sx, int sy, int sz,
			   int ex, int ey, int ez)
	: ix(sx), iy(sy), iz(sz), sx(sx), sy(sy), sz(sz),
	  ex(ex), ey(ey), ez(ez) {
    }
    friend class Region;
    int ix, iy, iz;
    int sx, sy, sz;
    int ex, ey, ez;
};

#endif
