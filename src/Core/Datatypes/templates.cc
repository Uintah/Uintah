/*
 * Manual template instantiations
 */


/*
 * These aren't used by Datatypes directly, but since they are used in
 * a lot of different modules, we instantiate them here to avoid bloat
 *
 * Find the bloaters with:
find . -name "*.ii" -print | xargs cat | sort | uniq -c | sort -nr | more
 */

#include <SCICore/Containers/LockingHandle.h>
#include <SCICore/Datatypes/GenSField.h>
#include <SCICore/Datatypes/FlatAttrib.h>
#include <SCICore/Datatypes/LatticeGeom.h>
#include <SCICore/Datatypes/AccelAttrib.h>
#include <SCICore/Datatypes/BrickAttrib.h>
#include <SCICore/Datatypes/IndexAttrib.h>
#include <SCICore/Datatypes/AnalytAttrib.h>

#include <SCICore/Datatypes/ScalarField.h>
using namespace SCICore::Datatypes;

#ifdef __sgi
#pragma set woff 1468
#endif

template class LockingHandle<ScalarField>;

#include <SCICore/Datatypes/ColumnMatrix.h>
template class LockingHandle<ColumnMatrix>;

#include <SCICore/Datatypes/Matrix.h>
template class LockingHandle<Matrix>;

#include <SCICore/Datatypes/Mesh.h>
template class LockingHandle<Mesh>;

#include <SCICore/Datatypes/Surface.h>
template class LockingHandle<Surface>;

template class DiscreteAttrib<double>;
template class FlatAttrib<double>;
template class AccelAttrib<double>;
template class BrickAttrib<double>;

template class AccelAttrib<unsigned char>;
template class IndexAttrib<double, unsigned char, AccelAttrib<unsigned char> >;

template class GenSField<double, LatticeGeom>;
template class GenSField<double, LatticeGeom, FlatAttrib<double> >;
template class GenSField<double, LatticeGeom, AccelAttrib<double> >;

template class AttribFunctor<double>;
template class AnalytAttrib<double>;

//template class MinMaxFunctor<double>;

#ifdef __sgi
#pragma reset woff 1468
#endif










