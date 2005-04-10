
#include <Uintah/Grid/FaceIterator.h>
#include <iostream>

using namespace Uintah;
using namespace std;

ostream& operator<<(ostream& out, const FaceIterator& c)
{
   out << "[FaceIterator at " << c.current() << " of " << c.end() << ']';
   return out;
}

//
// $Log$
// Revision 1.1  2000/06/14 21:59:35  jas
// Copied CCVariable stuff to make FCVariables.  Implementation is not
// correct for the actual data storage and iteration scheme.
//
//

