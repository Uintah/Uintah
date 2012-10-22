/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
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

#include <Core/Grid/Variables/Iterator.h>
#include <Core/Grid/Variables/GridIterator.h>
#include <Core/Grid/Variables/ListOfCellsIterator.h>
#include <Core/Grid/Variables/DifferenceIterator.h>
#include <Core/Grid/Variables/UnionIterator.h>

#include <iostream>
using namespace std;
using namespace Uintah;
int main()
{
  //create 3 iterators 1 that is the whole region and 2 that are the halves
  Iterator giter_big(GridIterator(IntVector(0,0,0), IntVector(2,2,2)));
  Iterator giter_left(GridIterator(IntVector(0,0,0),IntVector(2,2,1)));
  Iterator giter_right(GridIterator(IntVector(0,0,1),IntVector(2,2,2)));

  //add the 2 halves together to get the big iterator
  Iterator uiter(UnionIterator(giter_left,giter_right));

  //take the left away from the big iterator to get the right iterator
  Iterator diter(DifferenceIterator(giter_big,giter_left));

  //take the difference iterator away from the union iterator to get the left iterator
  Iterator diter2(DifferenceIterator(diter,uiter));

      
  //Compare big iterator and union iterator
  if(uiter!=giter_big)
  {
    cout << "Error union iterator not equal\n";
    return 1;
  }

  if(diter!=giter_right)
  {
    cout << "Error diffence iterator not equal\n";
    return 1;
  }
  
  if(diter2!=giter_left)
  {
    cout << "Error diffence iterator (2) not equal\n";
    return 1;
  }

  cout << "All tests passed\n";

  return 0;
}
