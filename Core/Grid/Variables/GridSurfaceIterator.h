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


#ifndef UINTAH_HOMEBREW_GridSurfaceIterator_H
#define UINTAH_HOMEBREW_GridSurfaceIterator_H

#include <Core/Geometry/IntVector.h>
#include <Core/Util/Assert.h>
#include <iterator>
#include <Core/Malloc/Allocator.h>
#include <Core/Grid/Variables/GridIterator.h>
namespace Uintah {

using SCIRun::IntVector;
 using std::ostream;

/**************************************

CLASS
   GridSurfaceIterator
   
   This iterator will iterate over cells that are on the 
   surface of a grid.  The user must specify the grid low and high 
   points along with an offset of the number of cells in each dimension 
   they wish the iterator to touch.  This offset can be positive for
   cells that are outside of the grid or negative for cells that are
   inside of the grid.  Each cell is touch exactly once.

   You can also specify the iterator region by specifying both the interior
   and exterior ranges.

GENERAL INFORMATION

   GridSurfaceIterator.h

   Justin Luitjens
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  

KEYWORDS
   GridSurfaceIterator

DESCRIPTION
   This iterator will iterate over cells that are on the 
   surface of a grid.  The user must specify the grid low and high 
   points along with an offset of the number of cells in each dimension 
   they wish the iterator to touch.  This offset can be positive for
   cells that are outside of the grid or negative for cells that are
   inside of the grid.  Each cell is touch exactly once.
   
   You can also specify the iterator region by specifying both the interior
   and exterior ranges.
  
WARNING
  
****************************************/

 class GridSurfaceIterator : 
 public BaseIterator {
   public:
     inline ~GridSurfaceIterator() {}

     //////////
     // Insert Documentation Here:
     inline void operator++(int) {
       this->operator++();
     }

     inline GridSurfaceIterator& operator++() {

       //increment current Face Iterator
       d_iter++;

       //increment face iterator if necessary
       while(d_iter.done() && d_curFace!=NULLFACE)
       {
         d_curFace=getNextFace(d_curFace);
         d_iter=getFaceIterator(d_curFace);
       }

       //set done flag
       d_done= d_curFace==NULLFACE || d_iter.done();

       //if(!d_done)
       // cout << " GSI face: " << d_curFace << " iter: " << d_iter << endl;
       return *this;
     }

     //////////
     // Insert Documentation Here:
     inline GridSurfaceIterator operator+=(int step) {
       GridSurfaceIterator old(*this);
       for (int i = 0; i < step; i++) 
         ++(*this);
       return old;
     }

     //////////
     // Insert Documentation Here:
     inline bool done() const {
       return d_done;
     }

     IntVector operator*() const {
       ASSERT(!d_done);
       return *d_iter;
     }
     inline GridSurfaceIterator(const IntVector& low, const IntVector& high, const IntVector& offset) {
       //compute interior and exterior points
       d_int_low=Max(low,low-offset);
       d_ext_low=Min(low,low-offset);
       d_int_high=Min(high,high+offset);
       d_ext_high=Max(high,high+offset);

       //cout << "Int: " << d_int_low << "->" << d_int_high << endl;
       //cout << "Ext: " << d_ext_low << "->" << d_ext_high << endl;
       reset();
     }
     inline GridSurfaceIterator(const IntVector& int_low, const IntVector& int_high, const IntVector& ext_low, const IntVector& ext_high)
       : d_int_low(int_low), d_int_high(int_high), d_ext_low(ext_low), d_ext_high(ext_high) {
         reset();
     }
     inline IntVector begin() const {
       return d_ext_low;
     }
     inline IntVector end() const {
       return d_ext_high;
     }
     /**
     * Return the number of cells in the iterator
     */
     inline unsigned int size() const
     {
      IntVector size_int=d_int_high-d_int_low;
      IntVector size_ext=d_ext_high-d_ext_low;

      return size_ext.x()*size_ext.y()*size_ext.z()-size_int.x()*size_int.y()*size_int.z();
     };
     inline GridSurfaceIterator(const GridSurfaceIterator& copy)
       : d_int_low(copy.d_int_low), d_int_high(copy.d_int_high), d_ext_low(copy.d_ext_low), d_ext_high(copy.d_ext_high), d_curFace(copy.d_curFace), d_iter(copy.d_iter), d_done(copy.d_done) {
       }

     inline GridSurfaceIterator& operator=( const GridSurfaceIterator& copy ) {
       d_int_low=copy.d_int_low;
       d_int_high=copy.d_int_high;
       d_ext_low=copy.d_ext_low;
       d_ext_high=copy.d_ext_high;
       d_iter=copy.d_iter;
       d_done = copy.d_done;
       return *this;
     }
     bool operator==(const GridSurfaceIterator& o) const
     {
       return d_int_low==o.d_int_low &&
         d_int_high==o.d_int_high &&
         d_ext_low==o.d_ext_low &&
         d_ext_high==o.d_ext_high &&
         *d_iter==*o.d_iter;
     }

     bool operator!=(const GridSurfaceIterator& o) const
     {
       return d_int_low!=o.d_int_low ||
         d_int_high!=o.d_int_high ||
         d_ext_low!=o.d_ext_low ||
         d_ext_high!=o.d_ext_high ||
         *d_iter!=*o.d_iter;
     }
     friend std::ostream& operator<<(std::ostream& out, const Uintah::GridSurfaceIterator& b);

     friend class GridIterator;

     inline void reset()
     {
       d_curFace=XMINUS;
       d_iter=getFaceIterator(d_curFace);
       //cout << "reset called xdone: " << d_iter.done() << endl;
       //increment face iterator if necessary
       while(d_iter.done() && d_curFace<NULLFACE)
       {
         //cout << "reset new face needed\n";
         d_curFace=getNextFace(d_curFace);
         d_iter=getFaceIterator(d_curFace);
       }

       d_done= d_curFace==NULLFACE || d_iter.done();
     }

     ostream& limits(ostream& out) const
     {
       out << begin() << " " << end() - IntVector(1,1,1);
       return out;
     }

   private:
     GridSurfaceIterator();

     GridSurfaceIterator* clone() const
     {
       return scinew GridSurfaceIterator(*this);
     }

     ostream& put(std::ostream& out) const
     {
       out << *this;
       return out;
     }

     enum Face {XMINUS=0,YMINUS=1,ZMINUS=2,XPLUS=3,YPLUS=4,ZPLUS=5,NULLFACE=6};

     Face getNextFace(const Face face)
     {
       switch(face)
       {
         case XMINUS:
           return YMINUS;
         case YMINUS:
           return ZMINUS;
         case ZMINUS:
           return XPLUS;
         case XPLUS:
           return YPLUS;
         case YPLUS:
           return ZPLUS;
         default:
           return NULLFACE;
       }
     }
     GridIterator getFaceIterator(Face curFace)
     {
       IntVector low,high;
       switch(curFace)
       {
         case XMINUS:
           low=d_ext_low;
           high=d_ext_high;
           //restrict to x face
           high.modifiable_x()=d_int_low.x();
           break;
         case YMINUS:
           low=d_ext_low;
           high=d_ext_high;
           //restrict to the y face
           high.modifiable_y()=d_int_low.y();
           //remove x face cells
           low.modifiable_x()=d_int_low.x();
           high.modifiable_x()=d_int_high.x();
           break;
         case ZMINUS:
           low=d_ext_low;
           high=d_ext_high;
           //restrict to the z face
           high.modifiable_z()=d_int_low.z();
           //remove x face cells
           low.modifiable_x()=d_int_low.x();
           high.modifiable_x()=d_int_high.x();
           //remove y face cells
           low.modifiable_y()=d_int_low.y();
           high.modifiable_y()=d_int_high.y();
           break;
         case XPLUS:
           low=d_ext_low;
           high=d_ext_high;
           //restrict to x face
           low.modifiable_x()=d_int_high.x();
           break;
         case YPLUS:
           low=d_ext_low;
           high=d_ext_high;
           //restrict to y face
           low.modifiable_y()=d_int_high.y();
           //remove x face cells
           low.modifiable_x()=d_int_low.x();
           high.modifiable_x()=d_int_high.x();
           break;
         case ZPLUS:
           low=d_ext_low;
           high=d_ext_high;
           //restrict to z face
           low.modifiable_z()=d_int_high.z();
           //remove x face cells
           low.modifiable_x()=d_int_low.x();
           high.modifiable_x()=d_int_high.x();
           //remove y face cells
           low.modifiable_y()=d_int_low.y();
           high.modifiable_y()=d_int_high.y();
           break;
         default:
           //return an empty iterator
           low=IntVector(0,0,0),high=IntVector(0,0,0);

       }
       //cout << "New Face Iterator, face: " << curFace << " low: " << low << " high: " << high << endl;
       return GridIterator(low,high);
     }

     //////////
     // Insert Documentation Here:
     IntVector d_int_low, d_int_high;
     IntVector d_ext_low, d_ext_high;
     Face d_curFace;            //The current face we are iterating over
     GridIterator d_iter;      //The current face iterator
     bool d_done;

}; // end class GridSurfaceIterator

} // End namespace Uintah
  
#endif
