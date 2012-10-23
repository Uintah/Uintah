/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

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
#include <Core/Malloc/Allocator.h>



using namespace SCIRun;

#include <Core/Geometry/Tensor.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Datatypes/Matrix.h>
#include <Core/Datatypes/NrrdData.h>
#include <Core/Persistent/PersistentSTL.h>
#include <Core/Datatypes/PropertyManager.h>
#include <Core/Datatypes/GenericField.h>
#include <vector>

#if !defined(__sgi)
// Needed for optimized linux build only
#  if 0
template void Pio<char, char>(Piostream&, pair<char, char>&);
template void Pio<int, int>(Piostream&, pair<int, int>&);
template void Pio<float, float>(Piostream&, pair<float, float>&);
template void Pio<int, double>(Piostream&, pair<int, double>&);
template void Pio<double, double>(Piostream&, pair<double, double>&);
template void Pio<short, short>(Piostream&, pair<short, short>&);
template void Pio<unsigned char, unsigned char>(Piostream&, pair<unsigned char,
		  unsigned char>&);
template void Pio<unsigned int, unsigned int>(Piostream&, pair<unsigned int,
		  unsigned int>&);
template void Pio<unsigned short, unsigned short>(Piostream&, pair<unsigned short,
		  unsigned short>&);
#  endif
#endif

template class LockingHandle<ColumnMatrix>;
template class LockingHandle<Matrix>;

//Index types
const TypeDescription* get_type_description(NodeIndex<int>*);
const TypeDescription* get_type_description(EdgeIndex<int>*);
const TypeDescription* get_type_description(FaceIndex<int>*);
const TypeDescription* get_type_description(CellIndex<int>*);

const TypeDescription* get_type_description(vector<NodeIndex<int> >*);
const TypeDescription* get_type_description(vector<EdgeIndex<int> >*);
const TypeDescription* get_type_description(vector<FaceIndex<int> >*);
const TypeDescription* get_type_description(vector<CellIndex<int> >*);

// Property types
template class Property<int>;
template class Property<string>;
template class Property<double>;
template class Property<float>;
template class Property<Array1<double> >;
template class Property<Array1<Tensor> >;
template class Property<pair<int,double> >;
template class Property<pair<double,double> >;
template class Property<pair<float,float> >;
template class Property<pair<unsigned int,unsigned int> >;
template class Property<pair<int,int> >;
template class Property<pair<unsigned short,unsigned short> >;
template class Property<pair<short,short> >;
template class Property<pair<unsigned char,unsigned char> >;
template class Property<pair<char,char> >;
template class Property<vector<pair<string,Tensor> > >;
template class Property<vector<pair<int,double> > >;
template class Property<LockingHandle<Matrix> >;
template class Property<LockingHandle<NrrdData> >;
template class Property<NodeIndex<unsigned int> >;

namespace SCIRun {

//Specializations from GenericField
template<>
void
load_partials(const vector<Vector> &grad, DenseMatrix &m)
{
  int i = 0;
  vector<Vector>::const_iterator iter = grad.begin();
  while(iter != grad.end()) {
    const Vector &v = *iter++;
    m.put(i, 0, v.x());
    m.put(i, 1, v.y());
    m.put(i, 2, v.z());
    ++i;
  }
}

template <>
void
load_partials(const vector<Tensor> &grad, DenseMatrix &m)
{
  ASSERTFAIL("unimplemented");
}


template <>
unsigned int
get_vsize(Vector*)
{
  return 3;
}

template <>
unsigned int
get_vsize(Tensor*)
{
  ASSERTFAIL("unimplemented");
}

} // end namespace SCIRun

