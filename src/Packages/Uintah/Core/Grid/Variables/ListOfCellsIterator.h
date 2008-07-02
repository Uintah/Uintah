
#ifndef UINTAH_HOMEBREW_ListOfCellsIterator_H
#define UINTAH_HOMEBREW_ListOfCellsIterator_H

#include <vector>
using namespace std;

#include <Core/Geometry/IntVector.h>

#include <Packages/Uintah/Core/Grid/uintahshare.h>
#include <Packages/Uintah/Core/Grid/Variables/BaseIterator.h>
namespace Uintah {

  using SCIRun::IntVector;

  /**************************************

    CLASS
    ListOfCellsIterator

    Base class for all iterators

    GENERAL INFORMATION

    ListOfCellsIterator.h

    Justin Luitjens
    Department of Computer Science
    University of Utah

    Center for the Simulation of Accidental Fires and Explosions (C-SAFE)

    Copyright (C) 2000 SCI Group

    KEYWORDS
    ListOfCellsIterator

    DESCRIPTION
    Base class for all iterators.

    WARNING

   ****************************************/

  class UINTAHSHARE ListOfCellsIterator : public BaseIterator {

    friend ostream& operator<<(std::ostream& out, const Uintah::ListOfCellsIterator& b);

    public:

    ListOfCellsIterator() : index_(0) {}

    ListOfCellsIterator(const ListOfCellsIterator &copy) : listOfCells_(copy.listOfCells_) { }

    /**
     * prefix operator to move the iterator forward
     */
    ListOfCellsIterator& operator++() {index_++; return *this;} 

    /**
     * postfix operator to move the iterator forward
     * does not return the iterator because of performance issues
     */
    void operator++(int) {index_++;} 

    /**
     * returns true if the iterator is done
     */    
    bool done() const { return index_==listOfCells_.size(); }

    /**
     * returns the IntVector that the current iterator is pointing at
     */
    IntVector operator*() const { ASSERT(index_<listOfCells_.size()); return listOfCells_[index_]; }

    /**
     * Return the first element of the iterator
     */
    inline IntVector begin() const { return listOfCells_.front(); }

    /**
     * Return the last element of the iterator
     */
    inline IntVector end() const { return listOfCells_.back(); }

    /**
     * adds a cell to the list of cells
     */
    inline void add(const IntVector& c) { listOfCells_.push_back(c); }

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
    ListOfCellsIterator* clone() const
    {
      return new ListOfCellsIterator(*this);

    };
    
    virtual ostream& put(std::ostream& out) const
    {
      out << *this;
      return out;
    }



    //vector to store cells
    vector<IntVector> listOfCells_;

    //index into the iterator
    unsigned int index_;

    }; // end class ListOfCellsIterator

} // End namespace Uintah
  
#endif
