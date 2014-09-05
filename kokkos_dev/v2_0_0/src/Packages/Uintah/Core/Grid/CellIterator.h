
#ifndef UINTAH_HOMEBREW_CellIterator_H
#define UINTAH_HOMEBREW_CellIterator_H

#include <Core/Geometry/IntVector.h>

namespace Uintah {
  using SCIRun::IntVector;

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
    inline void operator++(int) {
      // This does not return the old iterator due to performance problems
      // on some compilers...
      if(++d_cur.modifiable_x() >= d_e.x()){
	d_cur.modifiable_x() = d_s.x();
	if(++d_cur.modifiable_y() >= d_e.y()){
	  d_cur.modifiable_y() = d_s.y();
	  ++d_cur.modifiable_z();
	  if(d_cur.modifiable_z() >= d_e.z())
	    d_done=true;
	}
      }
    }

    inline CellIterator& operator++() {
      if(++d_cur.modifiable_x() >= d_e.x()){
	d_cur.modifiable_x() = d_s.x();
	if(++d_cur.modifiable_y() >= d_e.y()){
	  d_cur.modifiable_y() = d_s.y();
	  ++d_cur.modifiable_z();
	  if(d_cur.modifiable_z() >= d_e.z())
	    d_done=true;
	}
      }
      return *this;
    }
    
    //////////
    // Insert Documentation Here:
    inline CellIterator operator+=(int step) {
      CellIterator old(*this);
      
      for (int i = 0; i < step; i++) {
	if(++d_cur.modifiable_x() >= d_e.x()){
	  d_cur.modifiable_x() = d_s.x();
	  if(++d_cur.modifiable_y() >= d_e.y()){
	    d_cur.modifiable_y() = d_s.y();
	    ++d_cur.modifiable_z();
	    if(d_cur.modifiable_z() >= d_e.z())
	      d_done=true;
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
      return d_done;
    }
     
    const IntVector& operator*() const {
      ASSERT(!d_done);
      return d_cur;
    }
    inline CellIterator(const IntVector& s, const IntVector& e)
      : d_s(s), d_e(e), d_cur(s), d_done(false) {
    }
    inline IntVector begin() const {
      return d_s;
    }
    inline IntVector end() const {
      return d_e;
    }
    inline CellIterator(const CellIterator& copy)
      : d_s(copy.d_s), d_e(copy.d_e), d_cur(copy.d_cur), d_done(copy.d_done) {
    }
    
  private:
    CellIterator();
    CellIterator& operator=(const CellIterator& copy);
    
    //////////
    // Insert Documentation Here:
    IntVector d_s,d_e;
    IntVector d_cur;
    bool d_done;
  };
} // End namespace Uintah
  
std::ostream& operator<<(std::ostream& out, const Uintah::CellIterator& b);
 
#endif
