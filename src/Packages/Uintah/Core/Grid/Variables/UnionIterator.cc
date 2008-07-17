
#include <Packages/Uintah/Core/Grid/Variables/UnionIterator.h>

namespace Uintah
{

  UnionIterator::UnionIterator(Iterator iter1, Iterator iter2) : ListOfCellsIterator()
  {
    iter1.reset();
    iter2.reset();
    while(!iter1.done() && !iter2.done() )
    {
      if(*iter1==*iter2) //in both lists 
      {
        add(*iter1); //add to list
        //advance iterators
        iter1++;
        iter2++;
      }
      else if(compareIt(*iter1,*iter2)) //in iter1 only
      {
        //add to list
        add(*iter1);
        //advance iterator
        iter1++;
      }
      else    //in iter2 only
      {
        //add to list
        add(*iter2);
        //advance iterator
        iter2++;              
      }
    }

    //add remaining cells in iter1
    while(!iter1.done())
    {
      add(*iter1);
      iter1++;
    }     
    //add remaining cells in iter2
    while(!iter2.done())
    {
      add(*iter2);
      iter2++;
    }     
  }
  std::ostream& operator<<(std::ostream& out, const Uintah::UnionIterator& b)
  {
    out << "[UnionIterator at index" << b.index_ << " of " << b.listOfCells_.size() << "]" ;

    return out;

  }
}
  
