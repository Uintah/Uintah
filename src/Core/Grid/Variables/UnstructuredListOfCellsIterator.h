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


#ifndef UINTAH_HOMEBREW_UnstructuredListOfCellsIterator_H
#define UINTAH_HOMEBREW_UnstructuredListOfCellsIterator_H

#include <climits>
#include <vector>

#include <Core/Geometry/IntVector.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Grid/Variables/BaseIterator.h>
namespace Uintah {

  
  /**************************************

    CLASS
    UnstructuredListOfCellsIterator

    Base class for all iterators

    GENERAL INFORMATION

    UnstructuredListOfCellsIterator.h

    Justin Luitjens
    Department of Computer Science
    University of Utah

    Center for the Simulation of Accidental Fires and Explosions (C-SAFE)


    KEYWORDS
    UnstructuredListOfCellsIterator

    DESCRIPTION
    Base class for all iterators.

    WARNING

   ****************************************/

  class UnstructuredListOfCellsIterator : public BaseIterator {

    friend std::ostream& operator<<(std::ostream& out, const Uintah::UnstructuredListOfCellsIterator& b);

    public:

    UnstructuredListOfCellsIterator() : index_(0) { listOfCells_.push_back(IntVector(INT_MAX,INT_MAX,INT_MAX)); }

  UnstructuredListOfCellsIterator(const UnstructuredListOfCellsIterator &copy) : listOfCells_(copy.listOfCells_) {reset(); }

    /**
     * prefix operator to move the iterator forward
     */
    UnstructuredListOfCellsIterator& operator++() {index_++; return *this;} 

    /**
     * postfix operator to move the iterator forward
     * does not return the iterator because of performance issues
     */
    void operator++(int) {index_++;} 

    /**
     * returns true if the iterator is done
     */    
    bool done() const { return index_==listOfCells_.size()-1; }

    /**
     * returns the IntVector that the current iterator is pointing at
     */
    IntVector operator*() const { ASSERT(index_<listOfCells_.size()); return listOfCells_[index_]; }

    /**
     * Return the first element of the iterator
     */
    inline IntVector begin() const { return listOfCells_.front(); }

    /**
     * Return one past the last element of the iterator
     */
    inline IntVector end() const { return listOfCells_.back(); }
    
    /**
     * Return the number of cells in the iterator
     */
    inline unsigned int size() const {return listOfCells_.size()-1;};

    /**
     * adds a cell to the list of cells
     */
    inline void add(const IntVector& c) 
    {
      //place at back of list
      listOfCells_.back()=c;
      //readd sentinal to list
      listOfCells_.push_back(IntVector(INT_MAX,INT_MAX,INT_MAX));
    }


    /**
     * resets the iterator
     */
    inline void reset()
    {
      index_=0;
    }

    protected:
    /**
     * Returns a pointer to a deep copy of the virtual class
     * this should be used only by the Iterator class
     */
    UnstructuredListOfCellsIterator* clone() const
    {
      return scinew UnstructuredListOfCellsIterator(*this);

    };
    
    virtual std::ostream& put(std::ostream& out) const
    {
      out << *this;
      return out;
    }

    virtual std::ostream& limits(std::ostream& out) const
    {
      out << begin() << " " << end();
      return out;
    }

    //vector to store cells
    std::vector<IntVector> listOfCells_;

    //index into the iterator
    unsigned int index_;

    }; // end class UnstructuredListOfCellsIterator

} // End namespace Uintah
  
#endif
