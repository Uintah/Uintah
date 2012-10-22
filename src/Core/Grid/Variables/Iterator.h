/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
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


#ifndef UINTAH_HOMEBREW_Iterator_H
#define UINTAH_HOMEBREW_Iterator_H

#include <Core/Geometry/IntVector.h>
#include <Core/Util/Assert.h>
#include <Core/Grid/Variables/BaseIterator.h>

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


    KEYWORDS
    Iterator

    DESCRIPTION
    Base class for all iterators.

    WARNING

   ****************************************/

  class Iterator {
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
      Iterator(const Iterator& copy)
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
      * Return the number of cells in the iterator
      */
      inline unsigned int size() const 
      { 
        ASSERT(iter_!=NULL);  
        return iter_->size(); 
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

      std::ostream& limits(std::ostream& out) const 
        {
          return iter_->limits(out);
        }

    private:
      friend std::ostream& operator<<(std::ostream& out, 
                                      const Uintah::Iterator& b);
      
      inline std::ostream& put(std::ostream& out) const { iter_->put(out); return out;}
      
      //a pointer to the base class iterator
      BaseIterator *iter_;
      

  }; // end class Iterator

} // End namespace Uintah
  
#endif
