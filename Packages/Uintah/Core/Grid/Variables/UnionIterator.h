
#ifndef UINTAH_HOMEBREW_UnionIterator_H
#define UINTAH_HOMEBREW_UnionIterator_H

#include <Core/Geometry/IntVector.h>

#include <Packages/Uintah/Core/Grid/uintahshare.h>
#include <Packages/Uintah/Core/Grid/Variables/BaseIterator.h>
#include <Packages/Uintah/Core/Grid/Variables/Iterator.h>
#include <Packages/Uintah/Core/Grid/Variables/ListOfCellsIterator.h>

namespace Uintah {

  using SCIRun::IntVector;

  /**************************************

    CLASS
    UnionIterator

    This iterator will iterator over the union between two iterators

    GENERAL INFORMATION

    UnionIterator.h

    Justin Luitjens
    Department of Computer Science
    University of Utah

    Center for the Simulation of Accidental Fires and Explosions (C-SAFE)

    Copyright (C) 2000 SCI Group

    KEYWORDS
    UnionIterator

    DESCRIPTION
    This iterator will iterator over the union between two iterators

    WARNING

   ****************************************/

  class UINTAHSHARE UnionIterator : public ListOfCellsIterator {
    friend ostream& operator<<(std::ostream& out, const Uintah::UnionIterator& b);
    public:

    UnionIterator(Iterator iter1, Iterator iter2);


    ostream& put(std::ostream& out) const
    {
      out << *this;
      return out;
    }

    private:
    UnionIterator() : ListOfCellsIterator() {}

    }; // end class UnionIterator

} // End namespace Uintah
  
#endif
