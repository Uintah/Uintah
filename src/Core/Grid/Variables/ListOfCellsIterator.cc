/*
 * The MIT License
 *
 * Copyright (c) 1997-2025 The University of Utah
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


#include <Core/Grid/Variables/ListOfCellsIterator.h>

namespace Uintah
{

  std::ostream& operator<<(std::ostream& out, const Uintah::ListOfCellsIterator& b)
  {    
    int last = b.mySize-1;
    IntVector l_start (b.listOfCells_[0][0] ,b.listOfCells_[0][1] ,b.listOfCells_[0][2] );
    IntVector l_end (b.listOfCells_[last][0] ,b.listOfCells_[last][1] ,b.listOfCells_[last][2] );
    out << "[ListOfCellsIterator with " << b.mySize<< " elements, first cell in list: "<< l_start 
        << " last cell in list: " << l_end << "]";

    return out;

  }
}
  
