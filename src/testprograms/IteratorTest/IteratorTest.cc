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

#include <Core/Grid/Variables/Iterator.h>
#include <Core/Grid/Variables/GridIterator.h>
#include <Core/Grid/Variables/ListOfCellsIterator.h>
#include <Core/Grid/Variables/DifferenceIterator.h>
#include <Core/Grid/Variables/UnionIterator.h>
#include <Core/Parallel/KokkosTools.h>

#include <iostream>

int main()
{

#ifdef HAVE_KOKKOS
    Kokkos::initialize();
#endif //HAVE_KOKKOS

  //create 3 iterators 1 that is the whole region and 2 that are the halves
  Uintah::Iterator giter_big(Uintah::GridIterator(Uintah::IntVector(0,0,0), Uintah::IntVector(2,2,2)));
  Uintah::Iterator giter_left(Uintah::GridIterator(Uintah::IntVector(0,0,0),Uintah::IntVector(2,2,1)));
  Uintah::Iterator giter_right(Uintah::GridIterator(Uintah::IntVector(0,0,1),Uintah::IntVector(2,2,2)));

  //add the 2 halves together to get the big iterator
  Uintah::Iterator uiter(Uintah::UnionIterator(giter_left,giter_right));

  //take the left away from the big iterator to get the right iterator
  Uintah::Iterator diter(Uintah::DifferenceIterator(giter_big,giter_left));

  //take the difference iterator away from the union iterator to get the left iterator
  Uintah::Iterator diter2(Uintah::DifferenceIterator(diter,uiter));

      
  //Compare big iterator and union iterator
  if(uiter!=giter_big)
  {
    std::cout << "Error union iterator not equal\n";
    return 1;
  }

  if(diter!=giter_right)
  {
    std::cout << "Error diffence iterator not equal\n";
    return 1;
  }
  
  if(diter2!=giter_left)
  {
    std::cout << "Error diffence iterator (2) not equal\n";
    return 1;
  }

  std::cout << "All tests passed\n";

#ifdef HAVE_KOKKOS
  Uintah::cleanupKokkosTools();
  Kokkos::finalize();
#endif //HAVE_KOKKOS

  return 0;
}
