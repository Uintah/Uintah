/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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


#include <Core/Grid/Variables/DifferenceIterator.h>

namespace Uintah
{

  DifferenceIterator::DifferenceIterator(Iterator iter1, Iterator iter2) : ListOfCellsIterator()
  {
    iter1.reset();
    iter2.reset();
    while(!iter1.done() && !iter2.done() )
    {
      if(*iter1==*iter2) //in both lists advance iterators
      {
        iter1++;
        iter2++;
      }
      else if(*iter1<*iter2) //in iter1 only
      {
        add(*iter1);
        iter1++;
      }
      else    //in iter2 only
      {
        iter2++;              
      }
    }
    //add remaining cells in iter1
    while(!iter1.done())
    {
      add(*iter1);
      iter1++;
    }     
  }
  std::ostream& operator<<(std::ostream& out, const Uintah::DifferenceIterator& b)
  {
    out << "[DifferenceIterator at index" << b.index_ << " of " << b.listOfCells_.size() << "]" ;

    return out;

  }
}
  
