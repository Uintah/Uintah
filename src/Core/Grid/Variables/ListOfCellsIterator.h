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


#ifndef UINTAH_HOMEBREW_ListOfCellsIterator_H
#define UINTAH_HOMEBREW_ListOfCellsIterator_H

#include <climits>
#include <vector>

#include <Core/Geometry/IntVector.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Grid/Variables/BaseIterator.h>
#include <Core/Grid/Variables/Iterator.h>
#include <signal.h>

namespace Uintah {

  
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


    KEYWORDS
    ListOfCellsIterator

    DESCRIPTION
    Base class for all iterators.

    WARNING

   ****************************************/

  class ListOfCellsIterator : public BaseIterator {

    friend std::ostream& operator<<(std::ostream& out, const Uintah::ListOfCellsIterator& b);

    public:


    ListOfCellsIterator(int size) : mySize(0), index_(0), listOfCells_(size+1) { listOfCells_[mySize]=IntVector(INT_MAX,INT_MAX,INT_MAX);}

    ListOfCellsIterator(const ListOfCellsIterator &copy) :
                                                           mySize(copy.mySize),
                                                           index_(0),
                                                            listOfCells_(copy.listOfCells_)
                                                                   {   reset(); }

  //ListOfCellsIterator(const Iterator &copy){

    //int i=0;
    //for ( copy.reset(); !copy.done(); copy++){ // copy cells over, but make for sure there are not duplicates
      //listOfCells_[i]=*copy;
      //i++;
    //}
    //mySize=i;
    //reset();
  //}

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
    bool done() const { return index_==mySize; }

    /**
     * returns the IntVector that the current iterator is pointing at
     */
    IntVector operator*() const { ASSERT(index_<mySize); return listOfCells_[index_]; }

      /**
       * Assignment operator - this is expensive as we have to allocate new memory
       */
      inline Uintah::ListOfCellsIterator& operator=( Uintah::Iterator& copy ) 
      {
        //delete old iterator

       int i=0; 
       for (copy.reset(); !copy.done(); copy++) { // copy iterator into portable container
          
         listOfCells_[i]=(*copy);
         i++;
       }
       mySize=i;


        return *this;
      }
    /**
     * Return the first element of the iterator
     */
    inline IntVector begin() const { return listOfCells_[0]; }

    /**
     * Return one past the last element of the iterator
     */
    inline IntVector end() const { return listOfCells_[mySize]; }
    
    /**
     * Return the number of cells in the iterator
     */
    inline unsigned int size() const {return mySize;};

    /**
     * adds a cell to the list of cells
     */
    inline void add(const IntVector& c) 
    {
      //place at back of list
      listOfCells_[mySize]=c;
      mySize++; 
      //readd sentinal to list
      listOfCells_[mySize]=IntVector(INT_MAX,INT_MAX,INT_MAX);
    }


    /**
     * resets the iterator
     */
    inline void reset()
    {
      index_=0;
    }

    inline std::vector<IntVector>& get_ref_to_iterator(){
      return listOfCells_;
    }

    protected:
    /**
     * Returns a pointer to a deep copy of the virtual class
     * this should be used only by the Iterator class
     */
    ListOfCellsIterator* clone() const
    {
      return scinew ListOfCellsIterator(*this);

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

//#if defined( UINTAH_ENABLE_KOKKOS )
    //Kokkos::View<IntVector*> listOfCells_;
//#else
//#endif
    unsigned int mySize{0};
    //index into the iterator
    unsigned int index_{0};
    std::vector<IntVector> listOfCells_{};

    private:
     // This old constructor has a static size for portability reasons.  It should be avoided since .
  ListOfCellsIterator() : mySize(0), index_(0), listOfCells_(10000) { listOfCells_[mySize]=IntVector(INT_MAX,INT_MAX,INT_MAX);
            std::cout<< "Unsupported constructor, use at your own risk, in Core/Grid/Variables/ListOfCellsIterator.h \n";
         }

    }; // end class ListOfCellsIterator


} // End namespace Uintah
  
#endif
