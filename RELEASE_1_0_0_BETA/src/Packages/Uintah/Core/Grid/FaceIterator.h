
#ifndef UINTAH_HOMEBREW_FaceIterator_H
#define UINTAH_HOMEBREW_FaceIterator_H

#include <Packages/Uintah/Core/Grid/Array3Index.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Core/Geometry/IntVector.h>

namespace Uintah {

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
	 
	 if(++d_ix >= d_ex){
	    d_ix = d_sx;
	    if(++d_iy >= d_ey){
	       d_iy = d_sy;
	       ++d_iz;
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
} // End namespace Uintah

std::ostream& operator<<(std::ostream& out, const Uintah::FaceIterator& b);

#endif
