
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <iostream>

using namespace Uintah;
using namespace std;

ostream& operator<<(ostream& out, const CellIterator& c)
{
   out << "[CellIterator at " << *c << " of " << c.end() << ']';
   return out;
}


