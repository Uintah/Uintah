
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
	 
	 if(++d_iz >= d_ez){
	    d_iz = d_sz;
	    if(++d_iy >= d_ey){
	       d_iy = d_sy;
	       ++d_ix;
	    }
	 }
	 return old;
      }
      
      //////////
      // Insert Documentation Here:
      inline bool done() const {
	 return d_ix >= d_ex || d_iy >= d_ey || d_iz >= d_ez;
      }

      IntVector operator*() {
	 return IntVector(d_ix, d_iy, d_iz);
      }
      IntVector index() const {
	 return IntVector(d_ix, d_iy, d_iz);
      }
      inline NodeIterator(int ix, int iy, int iz, int ex, int ey, int ez)
	 : d_sx(ix), d_sy(iy), d_sz(iz), d_ix(ix), d_iy(iy), d_iz(iz), d_ex(ex), d_ey(ey), d_ez(ez) {
      }
      inline IntVector current() const {
	 return IntVector(d_ix, d_iy, d_iz);
      }
      inline IntVector begin() const {
	 return IntVector(d_sx, d_sy, d_sz);
      }
      inline IntVector end() const {
	 return IntVector(d_ex, d_ey, d_ez);
      }
      inline NodeIterator(const NodeIterator& copy)
	 : d_ix(copy.d_ix), d_iy(copy.d_iy), d_iz(copy.d_iz),
	   d_sx(copy.d_sx), d_sy(copy.d_sx), d_sz(copy.d_sz),
	   d_ex(copy.d_ex), d_ey(copy.d_ey), d_ez(copy.d_ez) {
      }

   private:
      NodeIterator();
      NodeIterator& operator=(const NodeIterator& copy);
      inline NodeIterator(const Region* region, int ix, int iy, int iz)
	 : d_sx(ix), d_sy(iy), d_sz(iz), d_ix(ix), d_iy(iy), d_iz(iz),
	   d_ex(region->getNNodes().x()), d_ey(region->getNNodes().y()), d_ez(region->getNNodes().z()) {
      }
      
      //////////
      // Insert Documentation Here:
      int     d_ix, d_iy, d_iz;
      int     d_sx, d_sy, d_sz;
      int     d_ex, d_ey, d_ez;
   };
   
} // end namespace Uintah

//
// $Log$
// Revision 1.8  2000/05/02 06:07:22  sparker
// Implemented more of DataWarehouse and SerialMPM
//
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
