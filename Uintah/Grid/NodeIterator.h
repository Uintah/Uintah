
#ifndef UINTAH_HOMEBREW_NodeIterator_H
#define UINTAH_HOMEBREW_NodeIterator_H

#include "Array3Index.h"
#include "Region.h"

namespace Uintah {
namespace Grid {

/**************************************

CLASS
   NodeIterator
   
   Short Description...

GENERAL INFORMATION

   NodeIterator.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   NodeIterator

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

class NodeIterator {
public:
    inline ~NodeIterator() {}

    //////////
    // Insert Documentation Here:
    inline NodeIterator operator++(int) {
	NodeIterator old(*this);
	if(++d_iz > d_region->d_nz){
	    d_iz=0;
	    if(++d_iy > d_region->d_ny){
		d_iy=0;
		if(++d_ix > d_region->d_nx){
		    d_iy=d_region->d_ny+1; d_iz=d_region->d_nz+1;
		}
	    }
	}
	return old;
    }

    //////////
    // Insert Documentation Here:
    inline Array3Index operator*() const {
	return Array3Index(d_ix, d_iy, d_iz);
    }

    //////////
    // Insert Documentation Here:
    inline bool operator!=(const NodeIterator& n) const {
	return d_ix != n.d_ix || d_iy != n.d_iy || d_iz != n.d_iz;
    }
private:
    NodeIterator();
    inline NodeIterator(const NodeIterator& copy)
	: d_region(copy.d_region),
	  d_ix(copy.d_ix), d_iy(copy.d_iy), d_iz(copy.d_iz) {
    }
    NodeIterator& operator=(const NodeIterator& copy);
    inline NodeIterator(const Region* region, int ix, int iy, int iz)
	: d_region(region),
	  d_ix(ix), d_iy(iy), d_iz(iz) {
    }
	
    friend class Region;

    //////////
    // Insert Documentation Here:
    const Region* d_region;
          int     d_ix, d_iy, d_iz;
};

} // end namespace Grid
} // end namespace Uintah

//
// $Log$
// Revision 1.2  2000/03/16 22:07:59  dav
// Added the beginnings of cocoon docs.  Added namespaces.  Did a few other coding standards updates too
//
//

#endif
