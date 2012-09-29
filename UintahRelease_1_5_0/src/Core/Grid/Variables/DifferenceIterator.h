/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */


#ifndef UINTAH_HOMEBREW_DifferenceIterator_H
#define UINTAH_HOMEBREW_DifferenceIterator_H

#include <Core/Geometry/IntVector.h>

#include <Core/Grid/Variables/BaseIterator.h>
#include <Core/Grid/Variables/Iterator.h>
#include <Core/Grid/Variables/ListOfCellsIterator.h>

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


    KEYWORDS
    DifferenceIterator

    DESCRIPTION
    This iterator will iterator over the difference between two iterators

    WARNING

   ****************************************/

  class DifferenceIterator : public ListOfCellsIterator {
    friend std::ostream& operator<<(std::ostream& out, const Uintah::DifferenceIterator& b);
    public:

    DifferenceIterator(Iterator iter1, Iterator iter2);


    std::ostream& put(std::ostream& out) const
    {
      out << *this;
      return out;
    }

    private:
    DifferenceIterator() : ListOfCellsIterator() {}

    }; // end class DifferenceIterator

} // End namespace Uintah
  
#endif
