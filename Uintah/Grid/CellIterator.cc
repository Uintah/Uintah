
#include <Uintah/Grid/CellIterator.h>

using namespace Uintah;
using namespace std;

ostream& operator<<(ostream& out, const CellIterator& c)
{
   out << "[CellIterator at " << c.current() << " of " << c.end() << ']';
   return out;
}

//
// $Log$
// Revision 1.1  2000/04/28 03:58:49  sparker
// Added CellIterator code
//
//

