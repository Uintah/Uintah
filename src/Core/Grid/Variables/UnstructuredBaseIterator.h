#ifndef UINTAH_UnstructuredBaseIterator_H
#define UINTAH_UnstructuredBaseIterator_H

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

#include <ostream>


namespace Uintah {

  /**************************************

    CLASS
    UnstructuredBaseIterator

    UnstructuredBase class for all iterators

    GENERAL INFORMATION

    UnstructuredBaseIterator.h

    Justin Luitjens
    Department of Computer Science
    University of Utah

    Center for the Simulation of Accidental Fires and Explosions (C-SAFE)


    KEYWORDS
    UnstructuredBaseIterator

    DESCRIPTION
    UnstructuredBase class for all iterators.  In Uintah an iterator is both a container
    and an iterator.  In order to be considered valid the iterator must visit
    the container in a sorted order from low to high (according to Uintah's 
    defined order, which is Z,Y,X).  In addition, each cell 
    can only be in the iterator once.  These conditions are not strictly 
    enforced and thus it is up to the creater of the iterator to enforce 
    these conditions.

    WARNING

   ****************************************/

  class UnstructuredBaseIterator : 
    public std::iterator<std::forward_iterator_tag, int> 
  {
    friend class Iterator;
    public:

    virtual ~UnstructuredBaseIterator() {}

    /**
     * prefix operator to move the iterator forward
     */
    virtual UnstructuredBaseIterator& operator++()=0; 

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
     * returns the int that the current iterator is pointing at
     */
    virtual int operator*() const=0;


    /**
     * Return the first element of the iterator
     */
    virtual int begin() const=0;

    /**
     * Return one past the last element of the iterator
     */
    virtual int end() const=0;

    /**
     * Return the number of cells in the iterator
     */
    virtual unsigned int size() const =0;

    virtual std::ostream& limits(std::ostream&) const = 0;


    protected:
    /**
     * Prevent this class from being instantiated, use an inherited class instead
     */
    UnstructuredBaseIterator() {};

    /**
     * Returns a pointer to a deep copy of the virtual class
     * this should be used only by the Iterator class
     */
    virtual UnstructuredBaseIterator* clone() const = 0;
   
    private:
    

    /**
     * send iterator information to the ostream 
     */
    virtual std::ostream& put(std::ostream&) const = 0;

    /**
     * resets the iterator to the begining
     */
    virtual void reset() = 0;

  }; // end class UnstructuredBaseIterator


} // End namespace Uintah
  
#endif
