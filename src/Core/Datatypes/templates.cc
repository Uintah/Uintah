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
#include <Core/Datatypes/GeneralField.h>
#include <Core/Datatypes/LatticeGeom.h>
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

//template class GeneralField<LatticeGeom>;
//template class GeneralField<LatticeGeom, FlatAttrib<double> >;
//template class GeneralField<LatticeGeom, AccelAttrib<double> >;

//template class AttribFunctor<double>;
//template class AnalytAttrib<double>;

//template class MinMaxFunctor<double>;

#include <Core/Datatypes/TypedFData.h>
#include <functional>
template class TypedFData<double>;
template class FData1D<double>;

class add5 : public std::binary_function<int &, double, void>
{
public:
  void operator()(int &result, double d) { result = int(d) + 5; }
};

//template class FDataUnOp<int, FData1D<double>, add5>;


class minwrap
{
public:
  void operator()(double &result, double a, double b) { result = Min(a, b); }
};

//template class FDataBinOp<double, FData1D<double>, FData1D<double>, minwrap>;

#ifdef __sgi
#pragma reset woff 1468
#endif










