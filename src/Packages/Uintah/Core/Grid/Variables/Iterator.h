
#ifndef UINTAH_HOMEBREW_Iterator_H
#define UINTAH_HOMEBREW_Iterator_H

#include <Core/Geometry/IntVector.h>
#include <Core/Util/Assert.h>
#include <Packages/Uintah/Core/Grid/uintahshare.h>
#include <Packages/Uintah/Core/Grid/Variables/BaseIterator.h>
using namespace std;
namespace Uintah {

  using SCIRun::IntVector;

  /**************************************

    CLASS
    Iterator

    A smart iterator that handles  calls to other iterators

    GENERAL INFORMATION

    Iterator.h

    Justin Luitjens
    Department of Computer Science
    University of Utah

    Center for the Simulation of Accidental Fires and Explosions (C-SAFE)

    Copyright (C) 2000 SCI Group

    KEYWORDS
    Iterator

    DESCRIPTION
    Base class for all iterators.

    WARNING

   ****************************************/

  class UINTAHSHARE Iterator {
    public:
      Iterator() : iter_(NULL) {}

      Iterator(const BaseIterator &it)
      {
        iter_= (&it)->clone();
      }

      ~Iterator()
      {
        if(iter_!=NULL)
        {
          delete iter_;
          iter_=NULL;
        }
      }
      Iterator(Iterator& copy)
      {
        //clone the new iterator (deep copy)
        iter_=copy.iter_->clone();
      }

      /**
       * prefix operator to move the iterator forward
       * no reference is returned for performance reasons
       */
      inline void operator++() 
      { 
        ASSERT(iter_!=NULL); 
        iter_->operator++(); 
      }

      /**
       * postfix operator to move the iterator forward
       */
      inline Iterator& operator++(int) 
      { 
        ASSERT(iter_!=NULL); 
        (*iter_)++; 
        return *this;
      }

      /**
       * returns true if the iterator is done
       */    
      inline bool done() const 
      { 
        ASSERT(iter_!=NULL); 
        return iter_->done(); 
      }

      /**
       * returns the IntVector that the current iterator is pointing at
       */
      inline IntVector operator*() const 
      {
        ASSERT(iter_!=NULL);  
        return **iter_; 
      }

      /**
       * Return the first element of the iterator
       */
      inline IntVector begin() const 
      { 
        ASSERT(iter_!=NULL);  
        return iter_->begin(); 
      }

      /**
       * Return the last element of the iterator
       */
      inline IntVector end() const 
      { 
        ASSERT(iter_!=NULL);  
        return iter_->end(); 
      }

      /**
       * Assignment operator - this is expensive as we have to allocate new memory
       */
      inline Iterator& operator=( const Iterator& copy ) 
      {
        //delete old iterator
        if(iter_!=NULL)
        {
          delete iter_;

        }

        //clone the new iterator (deep copy)
        iter_=copy.iter_->clone();

        return *this;
      }
      
      inline void reset()
      {
        iter_->reset();
      }
      
      bool operator==(const Iterator& b)
      {
        Iterator i1(*this);
        Iterator i2(*this);

        for(i1.reset(),i2.reset();!i1.done() && !i2.done();i1++,i2++)
        {
          if(*i1!=*i2)
          {
            return false;
          }
        }

        if(!i1.done() && !i2.done())
          return false;

        return true;
      }

      bool operator!=(const Iterator& b)
      {
        return !operator==(b);
      }

      ostream& limits(std::ostream& out) const 
        {
          return iter_->limits(out);
        }

    private:
      friend std::ostream& operator<<(std::ostream& out, 
                                      const Uintah::Iterator& b);
      
      inline ostream& put(ostream& out) const { iter_->put(out); return out;}
      
      //a pointer to the base class iterator
      BaseIterator *iter_;
      

  }; // end class Iterator

} // End namespace Uintah
  
#endif
