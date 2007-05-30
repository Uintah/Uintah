
#ifndef UINTAH_HOMEBREW_GridIterator_H
#define UINTAH_HOMEBREW_GridIterator_H

#include <SCIRun/Core/Geometry/IntVector.h>

#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/NodeIterator.h>

#include <Core/Grid/share.h>

#include <sgi_stl_warnings_off.h>
#include   <iostream>
#include <sgi_stl_warnings_on.h>

namespace Uintah {

using SCIRun::IntVector;

/**************************************

CLASS
   GridIterator
   
   Used to iterate over Cells and Nodes in a Grid.  (Will soon replace
   NodeIterator and CellIterator.)

GENERAL INFORMATION

   GridIterator.h

   Steven G. Parker
   J. Davison de St. Germain (Created based on CellIterator.h)
   SCI Institute
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2007 C-SAFE

KEYWORDS
   GridIterator

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

class SCISHARE GridIterator {
public:
  inline ~GridIterator() {}
    
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

  inline GridIterator& operator++() {
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
  inline GridIterator operator+=(int step) {
    GridIterator old(*this);
      
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
  inline GridIterator(const IntVector& s, const IntVector& e)
    : d_s(s), d_e(e), d_cur(s){
    if(d_s.x() >= d_e.x() || d_s.y() >= d_e.y() || d_s.z() >= d_e.z())
      d_done = true;
    else
      d_done = false;
  }
  inline IntVector begin() const {
    return d_s;
  }
  inline IntVector end() const {
    return d_e;
  }

  inline GridIterator( const GridIterator & copy ) :
    d_s(copy.d_s), d_e(copy.d_e), d_cur(copy.d_cur), d_done(copy.d_done)
  {
  }
  inline GridIterator( const CellIterator & copy ) :
    d_s(copy.d_s), d_e(copy.d_e), d_cur(copy.d_cur), d_done(copy.d_done)
  {
  }
  inline GridIterator( const NodeIterator & copy ) :
    d_s(copy.d_s), d_e(copy.d_e), d_cur( copy.d_ix, copy.d_iy, copy.d_iz ), d_done( copy.done() )
  {
  }
    
  inline GridIterator& operator=( const GridIterator& copy ) {
    d_s    = copy.d_s;
    d_e    = copy.d_e;
    d_cur  = copy.d_cur;
    d_done = copy.d_done;
    return *this;
  }

  friend std::ostream& operator<<(std::ostream& out, const Uintah::GridIterator& b);

private:
  GridIterator();
    
  //////////
  // Insert Documentation Here:
  IntVector d_s,d_e;
  IntVector d_cur;
  bool d_done;

}; // end class GridIterator

} // End namespace Uintah
  
#endif
