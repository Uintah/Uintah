
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
    inline CellIterator operator++(int) {
      CellIterator old(*this);
      
      if(++d_ix >= d_e.x()){
	d_ix = d_s.x();
	if(++d_iy >= d_e.y()){
	  d_iy = d_s.y();
	  ++d_iz;
	}
      }
      return old;
    }

    inline CellIterator& operator++() {
      if(++d_ix >= d_e.x()){
	d_ix = d_s.x();
	if(++d_iy >= d_e.y()){
	  d_iy = d_s.y();
	  ++d_iz;
	}
      }
      return *this;
    }
    
    //////////
    // Insert Documentation Here:
    inline CellIterator operator+=(int step) {
      CellIterator old(*this);
      
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
      : d_s(copy.d_s), d_e(copy.d_e),
	d_ix(copy.d_ix), d_iy(copy.d_iy), d_iz(copy.d_iz) {
    }
    
  private:
    CellIterator();
    CellIterator& operator=(const CellIterator& copy);
    
    //////////
    // Insert Documentation Here:
    IntVector d_s,d_e;
    int d_ix, d_iy, d_iz;
  };
} // End namespace Uintah
  
std::ostream& operator<<(std::ostream& out, const Uintah::CellIterator& b);

#endif
