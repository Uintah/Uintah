
#ifndef UINTAH_HOMEBREW_CellIterator_H
#define UINTAH_HOMEBREW_CellIterator_H

#include <Core/Geometry/IntVector.h>
#include <Core/Util/Assert.h>
#include <iterator>
#include <Core/Malloc/Allocator.h>
#include <Packages/Uintah/Core/Grid/uintahshare.h>
#include <Packages/Uintah/Core/Grid/Variables/BaseIterator.h>

namespace Uintah {

using SCIRun::IntVector;
 using std::ostream;

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

 class UINTAHSHARE CellIterator : 
 public BaseIterator {
   public:
     inline ~CellIterator() {}

     //////////
     // Insert Documentation Here:
     inline void operator++(int) {
       this->operator++();
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

     IntVector operator*() const {
       ASSERT(!d_done);
       return d_cur;
     }
     inline CellIterator(const IntVector& s, const IntVector& e)
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
     inline CellIterator(const CellIterator& copy)
       : d_s(copy.d_s), d_e(copy.d_e), d_cur(copy.d_cur), d_done(copy.d_done) {
       }

     inline CellIterator& operator=( const CellIterator& copy ) {
       d_s    = copy.d_s;
       d_e    = copy.d_e;
       d_cur  = copy.d_cur;
       d_done = copy.d_done;
       return *this;
     }
     bool operator==(const CellIterator& o) const
     {
       return begin()==o.begin() && end()==o.end() && d_cur==o.d_cur;
     }

     bool operator!=(const CellIterator& o) const
     {
       return begin()!=o.begin() || end()!=o.end() || d_cur!=o.d_cur;
     }
     friend std::ostream& operator<<(std::ostream& out, const Uintah::CellIterator& b);

     friend class GridIterator;

     inline void reset()
     {
       d_cur=d_s;
       d_done=false;
     }

     ostream& limits(ostream& out) const
     {
       out << begin() << " " << end() - IntVector(1,1,1);
       return out;
     }
   private:
     CellIterator();

     CellIterator* clone() const
     {
       return scinew CellIterator(*this);
     }

     ostream& put(ostream& out) const
     {
       out << *this;
       return out;
     }
     //////////
     // Insert Documentation Here:
     IntVector d_s,d_e;
     IntVector d_cur;
     bool d_done;

}; // end class CellIterator

} // End namespace Uintah
  
#endif
