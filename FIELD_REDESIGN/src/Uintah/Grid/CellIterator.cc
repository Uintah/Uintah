
#include <Uintah/Grid/CellIterator.h>
#include <iostream>

using namespace Uintah;
using namespace std;

ostream& operator<<(ostream& out, const CellIterator& c)
{
   out << "[CellIterator at " << c.current() << " of " << c.end() << ']';
   return out;
}

//
// $Log$
// Revision 1.2  2000/05/15 19:39:47  sparker
// Implemented initial version of DataArchive (output only so far)
// Other misc. cleanups
//
// Revision 1.1  2000/04/28 03:58:49  sparker
// Added CellIterator code
//
//

