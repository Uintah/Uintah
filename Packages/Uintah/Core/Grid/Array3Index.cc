
#include <Packages/Uintah/Core/Grid/Array3Index.h>
#include <iostream>

std::ostream& operator<<(std::ostream& out, const Array3Index& idx)
{
    out << "[" << idx.i() << ", " << idx.j() << ", " << idx.k() << "]";
    return out;
}

