
#ifndef UINTAH_HOMEBREW_DifferenceIterator_H
#define UINTAH_HOMEBREW_DifferenceIterator_H

#include <Core/Geometry/IntVector.h>

#include <Packages/Uintah/Core/Grid/uintahshare.h>
#include <Packages/Uintah/Core/Grid/Variables/BaseIterator.h>
#include <Packages/Uintah/Core/Grid/Variables/Iterator.h>
#include <Packages/Uintah/Core/Grid/Variables/ListOfCellsIterator.h>

namespace Uintah {

  using SCIRun::IntVector;

  /**************************************

    CLASS
    DifferenceIterator

    This iterator will iterator over the difference between two iterators

    GENERAL INFORMATION

    DifferenceIterator.h

    Justin Luitjens
    Department of Computer Science
    University of Utah

    Center for the Simulation of Accidental Fires and Explosions (C-SAFE)

    Copyright (C) 2000 SCI Group

    KEYWORDS
    DifferenceIterator

    DESCRIPTION
    This iterator will iterator over the difference between two iterators

    WARNING

   ****************************************/

  class UINTAHSHARE DifferenceIterator : public ListOfCellsIterator {
    friend ostream& operator<<(std::ostream& out, const Uintah::DifferenceIterator& b);
    public:

    DifferenceIterator(Iterator iter1, Iterator iter2);


    ostream& put(std::ostream& out) const
    {
      out << *this;
      return out;
    }

    private:
    DifferenceIterator() : ListOfCellsIterator() {}

    }; // end class DifferenceIterator

} // End namespace Uintah
  
#endif
