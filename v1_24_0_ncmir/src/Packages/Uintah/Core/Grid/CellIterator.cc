
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <sgi_stl_warnings_on.h>

using namespace Uintah;
using namespace std;

ostream& operator<<(ostream& out, const CellIterator& c)
{
   out << "[CellIterator at " << *c << " of " << c.end() << ']';
   return out;
}


