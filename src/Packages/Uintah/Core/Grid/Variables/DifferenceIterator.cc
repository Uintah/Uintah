
#include <Packages/Uintah/Core/Grid/Variables/DifferenceIterator.h>

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
    }     
  }
  std::ostream& operator<<(std::ostream& out, const Uintah::DifferenceIterator& b)
  {
    out << "[DifferenceIterator at index" << b.index_ << " of " << b.listOfCells_.size() << "]" ;

    return out;

  }
}
  
