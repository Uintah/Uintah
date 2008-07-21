
#include <Packages/Uintah/Core/Grid/Variables/GridIterator.h>



namespace Uintah {

ostream&
operator<<( ostream& out, const GridIterator& c )
{
   out << "[GridIterator at " << *c << " of " << c.end() << ']';
   return out;
}

} // end namespace Uintah


