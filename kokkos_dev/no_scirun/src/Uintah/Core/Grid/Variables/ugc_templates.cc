/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


#include <Core/Geometry/Vector.h>

#include <Uintah/Core/Grid/Variables/ParticleVariable.h>
#include <Uintah/Core/Grid/Variables/CCVariable.h>
#include <Uintah/Core/Grid/Variables/NCVariable.h>
#include <Uintah/Core/Grid/Variables/SFCXVariable.h>
#include <Uintah/Core/Grid/Variables/SFCYVariable.h>
#include <Uintah/Core/Grid/Variables/SFCZVariable.h>
#include <Uintah/Core/Grid/Variables/ReductionVariable.h>
#include <Uintah/Core/Grid/Variables/Stencil7.h>
#include <Uintah/Core/Math/Matrix3.h>
#include <Uintah/Core/Disclosure/TypeUtils.h>

#include <utility>

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#  pragma set woff 1468
#endif

template class Uintah::ParticleVariable<SCIRun::Vector>;
template class Uintah::ParticleVariable<Uintah::Matrix3>;
template class Uintah::ParticleVariable<SCIRun::Point>;
template class Uintah::ParticleVariable<double>;
template class Uintah::ParticleVariable<float>;
template class Uintah::ParticleVariable<int>;
//template class ParticleVariable<long int>;
template class Uintah::ParticleVariable<Uintah::long64>;

template class Uintah::NCVariable<SCIRun::Vector>;
template class Uintah::NCVariable<Uintah::Matrix3>;
template class Uintah::NCVariable<double>;
template class Uintah::NCVariable<float>;
template class Uintah::NCVariable<int>;
template class Uintah::NCVariable<Uintah::long64>;

template class Uintah::CCVariable<SCIRun::Vector>;
template class Uintah::CCVariable<Uintah::Matrix3>;
template class Uintah::CCVariable<double>;
template class Uintah::CCVariable<float>;
template class Uintah::CCVariable<int>;
template class Uintah::CCVariable<Uintah::long64>;
template class Uintah::CCVariable<Uintah::Stencil7>;

template class Uintah::SFCXVariable<SCIRun::Vector>;
template class Uintah::SFCXVariable<Uintah::Matrix3>;
template class Uintah::SFCXVariable<double>;
template class Uintah::SFCXVariable<float>;
template class Uintah::SFCXVariable<int>;
template class Uintah::SFCXVariable<Uintah::long64>;

template class Uintah::SFCYVariable<SCIRun::Vector>;
template class Uintah::SFCYVariable<Uintah::Matrix3>;
template class Uintah::SFCYVariable<double>;
template class Uintah::SFCYVariable<float>;
template class Uintah::SFCYVariable<int>;
template class Uintah::SFCYVariable<Uintah::long64>;

template class Uintah::SFCZVariable<SCIRun::Vector>;
template class Uintah::SFCZVariable<Uintah::Matrix3>;
template class Uintah::SFCZVariable<double>;
template class Uintah::SFCZVariable<float>;
template class Uintah::SFCZVariable<int>;
template class Uintah::SFCZVariable<Uintah::long64>;

template class Uintah::ReductionVariable<double, Uintah::Reductions::Min<double> >;

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#  pragma reset woff 1468
#endif
