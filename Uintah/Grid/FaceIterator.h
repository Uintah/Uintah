
#ifndef UINTAH_HOMEBREW_FaceIterator_H
#define UINTAH_HOMEBREW_FaceIterator_H

#include <Uintah/Grid/Array3Index.h>
#include <Uintah/Grid/Patch.h>
#include <SCICore/Geometry/IntVector.h>

namespace Uintah {
   using SCICore::Geometry::IntVector;

/**************************************

CLASS
   FaceIterator
   
   Short Description...

GENERAL INFORMATION

   FaceIterator.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   FaceIterator

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

   class FaceIterator {
   public:
      inline ~FaceIterator() {}
      
      //////////
      // Insert Documentation Here:
      inline FaceIterator operator++(int) {
	 FaceIterator old(*this);
	 
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
      inline FaceIterator(int ix, int iy, int iz, int ex, int ey, int ez)
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
      inline FaceIterator(const FaceIterator& copy)
	 : d_ix(copy.d_ix), d_iy(copy.d_iy), d_iz(copy.d_iz),
	   d_sx(copy.d_sx), d_sy(copy.d_sy), d_sz(copy.d_sz),
	   d_ex(copy.d_ex), d_ey(copy.d_ey), d_ez(copy.d_ez) {
      }

   private:
      FaceIterator();
      FaceIterator& operator=(const FaceIterator& copy);
      inline FaceIterator(const Patch* patch, int ix, int iy, int iz)
	 : d_sx(ix), d_sy(iy), d_sz(iz), d_ix(ix), d_iy(iy), d_iz(iz), d_ex(patch->getNFaces().x()), d_ey(patch->getNFaces().y()), d_ez(patch->getNFaces().z()) {
      }
      
      //////////
      // Insert Documentation Here:
      int     d_ix, d_iy, d_iz;
      int     d_sx, d_sy, d_sz;
      int     d_ex, d_ey, d_ez;
   };
   
} // end namespace Uintah

std::ostream& operator<<(std::ostream& out, const Uintah::FaceIterator& b);

//
// $Log$
// Revision 1.1  2000/06/14 21:59:35  jas
// Copied CCVariable stuff to make FCVariables.  Implementation is not
// correct for the actual data storage and iteration scheme.
//
//

#endif
