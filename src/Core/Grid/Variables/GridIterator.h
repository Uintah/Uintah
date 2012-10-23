/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */


#ifndef UINTAH_HOMEBREW_GridIterator_H
#define UINTAH_HOMEBREW_GridIterator_H

#include <Core/Geometry/IntVector.h>

#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/NodeIterator.h>


#include   <iostream>

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


    KEYWORDS
    GridIterator

    DESCRIPTION
    Long description...

    WARNING

   ****************************************/

  class GridIterator : public BaseIterator {
    friend ostream& operator<<( ostream& out,  const GridIterator& c );
    public:
    inline ~GridIterator() {}

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

    inline bool done() const {
      return d_done;
    }

    inline void reset() {
      d_done=d_s.x() >= d_e.x() || d_s.y() >= d_e.y() || d_s.z() >= d_e.z();
      d_cur=d_s;
    }

    IntVector operator*() const {
      ASSERT(!d_done);
      return d_cur;
    }

    inline GridIterator()
    {
      reset();
    }

    inline GridIterator(const IntVector& s, const IntVector& e)
      : d_s(s), d_e(e){

        reset();
      }
    inline IntVector begin() const {
      return d_s;
    }
    inline IntVector end() const {
      return d_e;
    }
    
    /**
     * Return the number of cells in the iterator
     */
    inline unsigned int size() const
    {
      IntVector size=d_e-d_s;
      if(size.x()<=0 || size.y()<=0 || size.z()<=0)
        return 0;
      else
        return size.x()*size.y()*size.z();
    };

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
      if (this == &copy)
        return *this;

      d_s    = copy.d_s;
      d_e    = copy.d_e;
      d_cur  = copy.d_cur;
      d_done = copy.d_done;
      return *this;
    }

    ostream& put(std::ostream& out) const
    {
      out << *this;
      return out;
    }

    ostream& limits(std::ostream& out) const
    {
      out << begin() << " " << end() - IntVector(1,1,1);
      return out;
    }

    private:
    /**
     * Returns a pointer to a deep copy of the virtual class
     * this should be used only by the Iterator class
     */
    GridIterator* clone() const {
      return scinew GridIterator(*this);
    }



    IntVector d_s,d_e;
    IntVector d_cur;
    bool d_done;

  }; // end class GridIterator

} // End namespace Uintah
  
#endif
