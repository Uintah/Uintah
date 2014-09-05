
#include <Core/Grid/BoundaryConditions/BoundCond.h>

#include <iostream>

namespace Uintah {

template<> void BoundCond<std::string>::debug() const { std::cout << "BoundCond: " << this << " - string\n"; }
template<> void BoundCond<double>::debug() const { std::cout << "BoundCond: " << this << " - double\n"; }
template<> void BoundCond<SCIRun::Vector>::debug() const { std::cout << "BoundCond: " << this << " - Vector\n"; }

}; // end namespace Uintah
