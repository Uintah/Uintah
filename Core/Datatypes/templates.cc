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

#include <Core/Containers/LockingHandle.h>
#include <Core/Datatypes/GenSField.h>
#include <Core/Datatypes/FlatAttrib.h>
#include <Core/Datatypes/LatticeGeom.h>
#include <Core/Datatypes/AccelAttrib.h>
#include <Core/Datatypes/BrickAttrib.h>
#include <Core/Datatypes/IndexAttrib.h>
#include <Core/Datatypes/AnalytAttrib.h>
#include <Core/Datatypes/ScalarField.h>

using namespace SCIRun;
#ifdef __sgi
#pragma set woff 1468
#endif

template class LockingHandle<ScalarField>;

#include <Core/Datatypes/ColumnMatrix.h>
template class LockingHandle<ColumnMatrix>;

#include <Core/Datatypes/Matrix.h>
template class LockingHandle<Matrix>;

#include <Core/Datatypes/Mesh.h>
template class LockingHandle<Mesh>;

#include <Core/Datatypes/Surface.h>
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










