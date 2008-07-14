#include <Packages/Uintah/Core/Grid/Variables/Iterator.h>
#include <Packages/Uintah/Core/Grid/Variables/GridIterator.h>
#include <Packages/Uintah/Core/Grid/Variables/ListOfCellsIterator.h>
#include <Packages/Uintah/Core/Grid/Variables/DifferenceIterator.h>
#include <Packages/Uintah/Core/Grid/Variables/UnionIterator.h>

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
