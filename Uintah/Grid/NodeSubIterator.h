#ifndef UINTAH_HOMEBREW_NodeSubIterator_H
#define UINTAH_HOMEBREW_NodeSubIterator_H

#include "Array3Index.h"

namespace Uintah {

/**************************************

CLASS
   NodeSubIterator
   
   Short Description...

GENERAL INFORMATION

   NodeSubIterator.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   NodeSubIterator

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

   class NodeSubIterator {
   public:
      inline NodeSubIterator() {
      }
      inline ~NodeSubIterator() {
      }
      
      //////////
      // Insert Documentation Here:
      inline NodeSubIterator operator++(int) {
	 NodeSubIterator old(*this);
	 if(++d_iz >= d_ez){
	    d_iz=d_sz;
	    if(++d_iy >= d_ey){
	       d_iy=d_sy;
	       if(++d_ix >= d_ex){
		  d_iy=d_ey; d_iz=d_ez;
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
      inline bool operator!=(const NodeSubIterator& n) const {
	 return d_ix != n.d_ix || d_iy != n.d_iy || d_iz != n.d_iz;
      }
      
      //////////
      // Insert Documentation Here:
      inline int x() const {
	 return d_ix;
      }
      
      //////////
      // Insert Documentation Here:
      inline int y() const {
	 return d_iy;
      }
      
      //////////
      // Insert Documentation Here:
      inline int z() const {
	 return d_iz;
      }
   private:
      inline NodeSubIterator(const NodeSubIterator& copy)
	 : d_ix(copy.d_ix), d_iy(copy.d_iy), d_iz(copy.d_iz),
	   d_sx(copy.d_sx), d_sy(copy.d_sy), d_sz(copy.d_sz),
	   d_ex(copy.d_ex), d_ey(copy.d_ey), d_ez(copy.d_ez)
      {
      }
      inline NodeSubIterator& operator=(const NodeSubIterator& copy) {
	 d_ix = copy.d_ix; d_iy = copy.d_iy; d_iz = copy.d_iz;
	 d_sx = copy.d_sx; d_sy = copy.d_sy; d_sz = copy.d_sz;
	 d_ex = copy.d_ex; d_ey = copy.d_ey; d_ez = copy.d_ez;
	 return *this;
      }
      inline NodeSubIterator(int sx, int sy, int sz,
			     int ex, int ey, int ez)
	 : d_ix(sx), d_iy(sy), d_iz(sz), d_sx(sx), d_sy(sy), d_sz(sz),
	   d_ex(ex), d_ey(ey), d_ez(ez) {
      }
      friend class Region;
      
      //////////
      // Insert Documentation Here:
      int d_ix, d_iy, d_iz;
      int d_sx, d_sy, d_sz;
      int d_ex, d_ey, d_ez;
   };
   
} // end namespace Uintah

//
// $Log$
// Revision 1.3  2000/04/26 06:48:50  sparker
// Streamlined namespaces
//
// Revision 1.2  2000/03/16 22:07:59  dav
// Added the beginnings of cocoon docs.  Added namespaces.  Did a few other coding standards updates too
//
//

#endif
