
#ifndef UINTAH_HOMEBREW_NodeIterator_H
#define UINTAH_HOMEBREW_NodeIterator_H

#include <Uintah/Grid/Array3Index.h>
#include <Uintah/Grid/Region.h>

namespace Uintah {

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
	 
	 if(++d_iz > d_region->d_box.upper().z()){
	    d_iz = d_region->d_box.lower().z();
	    if(++d_iy > d_region->d_box.upper().y()){
	       d_iy = d_region->d_box.lower().y();
	       if(++d_ix > d_region->d_box.upper().x()){
		  d_ix = d_region->d_box.upper().x() + 1;
		  d_iy = d_region->d_box.upper().y() + 1;
		  d_iz = d_region->d_box.upper().z() + 1;
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
      inline NodeIterator(const NodeIterator& copy)
	 : d_region(copy.d_region),
	   d_ix(copy.d_ix), d_iy(copy.d_iy), d_iz(copy.d_iz) {
      }

   private:
      NodeIterator();
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
   
} // end namespace Uintah

//
// $Log$
// Revision 1.7  2000/04/28 20:24:43  jas
// Moved some private copy constructors to public for linux.  Velocity
// field is now set from the input file.  Simulation state now correctly
// determines number of velocity fields.
//
// Revision 1.6  2000/04/26 06:48:50  sparker
// Streamlined namespaces
//
// Revision 1.5  2000/04/25 00:41:21  dav
// more changes to fix compilations
//
// Revision 1.4  2000/04/12 23:00:48  sparker
// Starting problem setup code
// Other compilation fixes
//
// Revision 1.3  2000/03/21 01:29:42  dav
// working to make MPM stuff compile successfully
//
// Revision 1.2  2000/03/16 22:07:59  dav
// Added the beginnings of cocoon docs.  Added namespaces.  Did a few other coding standards updates too
//
//

#endif
