/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/



#ifndef UINTAH_HOMEBREW_NodeIterator_H
#define UINTAH_HOMEBREW_NodeIterator_H

#include <Core/Geometry/IntVector.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Grid/Variables/BaseIterator.h>

namespace Uintah {

  using SCIRun::IntVector;

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

  class NodeIterator  :
    public BaseIterator {
      public:
        inline ~NodeIterator() {}

        //////////
        // Insert Documentation Here:
        inline NodeIterator& operator++() {

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
        inline void operator++(int) {
          this->operator++();
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

        IntVector operator*() const {
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
        inline NodeIterator(const NodeIterator& copy)
          : d_s(copy.d_s), d_e(copy.d_e),
          d_ix(copy.d_ix), d_iy(copy.d_iy), d_iz(copy.d_iz) {
          }

        friend class GridIterator;
        
        inline NodeIterator& operator=( const NodeIterator& copy ) {
          d_s   = copy.d_s;
          d_e   = copy.d_e;
          d_ix  = copy.d_ix;
          d_iy  = copy.d_iy;
          d_iz  = copy.d_iz;
          return *this;
        }

        bool operator==(const NodeIterator& o) const
        {
          return begin()==o.begin() && end()==o.end() && index()==o.index();
        }

        bool operator!=(const NodeIterator& o) const
        {
          return begin()!=o.begin() || end()!=o.end() || index()!=o.index();
        }

        inline void reset()
        {
          d_ix=d_s.x();
          d_iy=d_s.y();
          d_iz=d_s.z();
        }

        std::ostream& limits(std::ostream& out) const
        {
          out << begin() << " " << end() - IntVector(1,1,1);
          return out;
        }

      private:
        NodeIterator();

        NodeIterator* clone() const
        {
          return scinew NodeIterator(*this);
        }
        
        std::ostream& put(std::ostream& out) const
        {
          out << *this;
          return out;
        }

        //////////
        // Insert Documentation Here:
        IntVector d_s,d_e;
        int d_ix, d_iy, d_iz;
        
        friend std::ostream& operator<<(std::ostream& out, const Uintah::NodeIterator& b);
    }; // end class NodeIterator

} // End namespace Uintah

#endif
