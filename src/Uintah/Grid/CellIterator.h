
#ifndef UINTAH_HOMEBREW_CellIterator_H
#define UINTAH_HOMEBREW_CellIterator_H

#include <Uintah/Grid/Array3Index.h>
#include <Uintah/Grid/Patch.h>
#include <SCICore/Geometry/IntVector.h>

namespace Uintah {
   using SCICore::Geometry::IntVector;

/**************************************

CLASS
   CellIterator
   
   Short Description...

GENERAL INFORMATION

   CellIterator.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   CellIterator

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

   class CellIterator {
   public:
      inline ~CellIterator() {}
      
      //////////
      // Insert Documentation Here:
      inline CellIterator operator++(int) {
	 CellIterator old(*this);
	 
	 if(++d_iz >= d_e.z()){
	    d_iz = d_s.z();
	    if(++d_iy >= d_e.y()){
	       d_iy = d_s.y();
	       ++d_ix;
	    }
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
      inline CellIterator(const IntVector& s, const IntVector& e)
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
      inline CellIterator(const CellIterator& copy)
	 : d_ix(copy.d_ix), d_iy(copy.d_iy), d_iz(copy.d_iz),
	   d_s(copy.d_s), d_e(copy.d_e) {
      }

   private:
      CellIterator();
      CellIterator& operator=(const CellIterator& copy);
      
      //////////
      // Insert Documentation Here:
      IntVector d_s,d_e;
      int d_ix, d_iy, d_iz;
   };
   
} // end namespace Uintah

std::ostream& operator<<(std::ostream& out, const Uintah::CellIterator& b);

//
// $Log$
// Revision 1.6  2000/06/15 21:57:16  sparker
// Added multi-patch support (bugzilla #107)
// Changed interface to datawarehouse for particle data
// Particles now move from patch to patch
//
// Revision 1.5  2000/06/13 21:28:30  jas
// Added missing TypeUtils.h for fun_forgottherestofname and copy constructor
// was wrong for CellIterator.
//
// Revision 1.4  2000/05/30 20:19:28  sparker
// Changed new to scinew to help track down memory leaks
// Changed region to patch
//
// Revision 1.3  2000/04/28 20:24:43  jas
// Moved some private copy constructors to public for linux.  Velocity
// field is now set from the input file.  Simulation state now correctly
// determines number of velocity fields.
//
// Revision 1.2  2000/04/28 03:58:20  sparker
// Fixed countParticles
// Implemented createParticles, which doesn't quite work yet because the
//   data warehouse isn't there yet.
// Reduced the number of particles in the bar problem so that it will run
//   quickly during development cycles
//
// Revision 1.1  2000/04/27 23:19:10  sparker
// Object to iterator over cells
//
//

#endif
