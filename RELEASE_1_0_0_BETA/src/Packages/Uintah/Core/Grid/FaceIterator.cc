
#include <Packages/Uintah/Core/Grid/FaceIterator.h>
#include <iostream>

using namespace Uintah;
using namespace std;

ostream& operator<<(ostream& out, const FaceIterator& c)
{
   out << "[FaceIterator at " << c.current() << " of " << c.end() << ']';
   return out;
}


