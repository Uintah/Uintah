
#ifndef UINTAH_HOMEBREW_NodeIterator_H
#define UINTAH_HOMEBREW_NodeIterator_H

#include <Uintah/Grid/Array3Index.h>
#include <Uintah/Grid/Patch.h>

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
      
      if(++d_ix >= d_e.x()){
	d_ix = d_s.x();
	if(++d_iy >= d_e.y()){
	  d_iy = d_s.y();
	  ++d_iz;
	}
      }
      return old;
    }
    
    //////////
    // Insert Documentation Here:
    inline NodeIterator operator+=(int step) {
      NodeIterator old(*this);

      for (int i = 0; i < step; i++) {
	if(++d_ix >= d_e.x()){
	  d_ix = d_s.x();
	  if(++d_iy >= d_e.y()){
	    d_iy = d_s.y();
	    ++d_iz;
	  }
	}
	if (done())
	  break;
      }
      return old;
    }
    
    //////////
    // Insert Documentation Here:
    inline bool done() const {
      return d_ix >= d_e.x() || d_iy >= d_e.y() || d_iz >= d_e.z();
    }
    
    IntVector operator*() {
      return IntVector(d_ix, d_iy, d_iz);
    }
    IntVector index() const {
      return IntVector(d_ix, d_iy, d_iz);
    }
    inline NodeIterator(const IntVector& s, const IntVector& e)
      : d_s(s), d_e(e) {
	d_ix = s.x();
	d_iy = s.y();
	d_iz = s.z();
    }
    inline IntVector current() const {
      return IntVector(d_ix, d_iy, d_iz);
    }
    inline IntVector begin() const {
      return d_s;
    }
    inline IntVector end() const {
      return d_e;
    }
    inline NodeIterator(const NodeIterator& copy)
      : d_s(copy.d_s), d_e(copy.d_e),
	d_ix(copy.d_ix), d_iy(copy.d_iy), d_iz(copy.d_iz) {
    }
    
  private:
    NodeIterator();
    NodeIterator& operator=(const NodeIterator& copy);
    
    //////////
    // Insert Documentation Here:
    IntVector d_s,d_e;
    int d_ix, d_iy, d_iz;
  };
} // end namespace Uintah

//
// $Log$
// Revision 1.14  2000/09/25 20:37:42  sparker
// Quiet g++ compiler warnings
// Work around g++ compiler bug instantiating vector<NCVariable<Vector> >
// Added computeVariableExtents to (eventually) simplify data warehouses
//
// Revision 1.13  2000/06/26 21:28:10  bigler
// Added += opporator.
//
// Revision 1.12  2000/06/16 05:19:21  sparker
// Changed arrays to fortran order
//
// Revision 1.11  2000/06/15 21:57:17  sparker
// Added multi-patch support (bugzilla #107)
// Changed interface to datawarehouse for particle data
// Particles now move from patch to patch
//
// Revision 1.10  2000/05/30 20:19:30  sparker
// Changed new to scinew to help track down memory leaks
// Changed region to patch
//
// Revision 1.9  2000/05/10 20:03:00  sparker
// Added support for ghost cells on node variables and particle variables
//  (work for 1 patch but not debugged for multiple)
// Do not schedule fracture tasks if fracture not enabled
// Added fracture directory to MPM sub.mk
// Be more uniform about using IntVector
// Made patches have a single uniform index space - still needs work
//
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
