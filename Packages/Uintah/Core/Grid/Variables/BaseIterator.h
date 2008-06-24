
#ifndef UINTAH_HOMEBREW_BaseIterator_H
#define UINTAH_HOMEBREW_BaseIterator_H

#include <ostream>
using namespace std;

#include <Core/Geometry/IntVector.h>

#include <Packages/Uintah/Core/Grid/uintahshare.h>
namespace Uintah {

  using SCIRun::IntVector;

  /**************************************

    CLASS
    BaseIterator

    Base class for all iterators

    GENERAL INFORMATION

    BaseIterator.h

    Justin Luitjens
    Department of Computer Science
    University of Utah

    Center for the Simulation of Accidental Fires and Explosions (C-SAFE)

    Copyright (C) 2000 SCI Group

    KEYWORDS
    BaseIterator

    DESCRIPTION
    Base class for all iterators.

    WARNING

   ****************************************/

  class UINTAHSHARE BaseIterator : 
    public std::iterator<std::forward_iterator_tag, IntVector> {
      friend class Iterator;
      public:

      virtual ~BaseIterator() {}
      
      /**
       * prefix operator to move the iterator forward
       */
      virtual BaseIterator& operator++()=0; 
      
      /**
       * postfix operator to move the iterator forward
       * does not return the iterator because of performance issues
       */
      virtual void operator++(int) = 0;
      
      /**
       * returns true if the iterator is done
       */    
      virtual bool done() const = 0;

      /**
       * returns the IntVector that the current iterator is pointing at
       */
      virtual IntVector operator*() const=0;

      /**
       * Return the first element of the iterator
       */
      virtual IntVector begin() const=0;

      /**
       * Return the last element of the iterator
       */
      virtual IntVector end() const=0;

      protected:
      /**
       * Prevent this class from being instantiated, use an inherited class instead
       */
      BaseIterator() {};

      private:
      /**
       * Returns a pointer to a deep copy of the virtual class
       * this should be used only by the Iterator class
       */
      virtual BaseIterator* clone() const = 0;
      
      /**
       * send iterator information to the ostream 
       */
      virtual ostream& put(ostream&) const = 0;

    }; // end class BaseIterator

} // End namespace Uintah
  
#endif
