
#include <Packages/Uintah/Core/Grid/Variables/ListOfCellsIterator.h>

namespace Uintah
{

  std::ostream& operator<<(std::ostream& out, const Uintah::ListOfCellsIterator& b)
  {
    out << "[ListOfCellsIterator at index" << b.index_ << " of " << b.listOfCells_.size() << "]" ;

    return out;

  }
}
  
