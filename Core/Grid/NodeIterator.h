
#ifndef UINTAH_HOMEBREW_NodeIterator_H
#define UINTAH_HOMEBREW_NodeIterator_H

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

} // End namespace Uintah

#endif
