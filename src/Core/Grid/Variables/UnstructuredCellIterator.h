#ifndef UINTAH_HOMEBREW_UnstructuredCellIterator_H
#define UINTAH_HOMEBREW_UnstructuredCellIterator_H

/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
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

#include <Core/Util/Assert.h>
#include <iterator>
#include <Core/Malloc/Allocator.h>
#include <Core/Grid/Variables/UnstructuredBaseIterator.h>

namespace Uintah {

/**************************************

CLASS
   UnstructuredCellIterator
   
   Short Description...

GENERAL INFORMATION

   UnstructuredCellIterator.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  

KEYWORDS
   UnstructuredCellIterator

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

 class UnstructuredCellIterator : 
 public UnstructuredBaseIterator {
   public:
     inline ~UnstructuredCellIterator() {}

     //////////
     // Insert Documentation Here:
     inline void operator++(int) {
       this->operator++();
     }

     inline UnstructuredCellIterator& operator++() {
       if(++d_cur >= d_e){
         d_cur = d_s;
	 d_done=true;
       }
       return *this;
     }


     //////////
     // Insert Documentation Here:
     inline UnstructuredCellIterator operator+=(int step) {
       UnstructuredCellIterator old(*this);

       for (int i = 0; i < step; i++) {
         if(++d_cur >= d_e){
           d_cur = d_s;
	   d_done=true;
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

     int operator*() const {
       ASSERT(!d_done);
       return d_cur;
     }
     inline UnstructuredCellIterator(const int& s, const int& e)
       : d_s(s), d_e(e){
       reset();
     }
     inline int begin() const {
       return d_s;
     }
     inline int end() const {
       return d_e;
     }
     /**
     * Return the number of cells in the iterator
     */
     inline unsigned int size() const
     {
       int size=d_e-d_s;
       if(size<=0)
         return 0;
       else
         return size;
     };
     inline UnstructuredCellIterator(const UnstructuredCellIterator& copy)
       : d_s(copy.d_s), d_e(copy.d_e), d_cur(copy.d_cur), d_done(copy.d_done) {
       }

     inline UnstructuredCellIterator& operator=( const UnstructuredCellIterator& copy ) {
       d_s    = copy.d_s;
       d_e    = copy.d_e;
       d_cur  = copy.d_cur;
       d_done = copy.d_done;
       return *this;
     }
     bool operator==(const UnstructuredCellIterator& o) const
     {
       return begin()==o.begin() && end()==o.end() && d_cur==o.d_cur;
     }

     bool operator!=(const UnstructuredCellIterator& o) const
     {
       return begin()!=o.begin() || end()!=o.end() || d_cur!=o.d_cur;
     }
     friend std::ostream& operator<<( std::ostream& out, const UnstructuredCellIterator& b );

     friend class GridIterator;
     friend class UnstructuredGridIterator;

     inline void reset()
     {
       d_cur=d_s;
       d_done=d_s >= d_e;
     }

     std::ostream& limits(std::ostream& out) const
     {
       out << begin() << " " << end() - 1;
       return out;
     }
   private:
     UnstructuredCellIterator();

     UnstructuredCellIterator* clone() const
     {
       return scinew UnstructuredCellIterator(*this);
     }

     std::ostream& put(std::ostream& out) const
     {
       out << *this;
       return out;
     }
     //////////
     // Insert Documentation Here:
     int d_s,d_e;
     int d_cur;
     bool d_done;

}; // end class UnstructuredCellIterator

} // End namespace Uintah
  
#endif
