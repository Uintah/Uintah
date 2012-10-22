/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
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
 *  MeshTet.h: Templated Meshs defined on a 3D Regular Grid
 *
 *  Written by:
 *   Marty Cole
 *   Department of Computer Science
 *   University of Utah
 *   January 2001
 *
 */

#ifndef Datatypes_FieldIndex_h
#define Datatypes_FieldIndex_h

#include <Core/Malloc/Allocator.h>
#include <Core/Persistent/Persistent.h>
#include <Core/Util/TypeDescription.h>
#include <Core/Datatypes/TypeName.h>
#include <vector>

namespace SCIRun {

using std::vector;

//! Base type for index types.
template <class T>
struct FieldIndexBase {
  typedef T value_type;
  
  FieldIndexBase(T i) :
    index_(i) {}

  //! Required interface for an Index.
  operator T const &() const { return index_; }

  std::ostream& str_render(std::ostream& os) const {
    os << index_;
    return os;
  }
  
  T index_;
};



//! Distinct type for node index.
template <class T>
struct NodeIndex : public FieldIndexBase<T> {
  NodeIndex() :
    FieldIndexBase<T>(0) {}
  NodeIndex(T index) :
    FieldIndexBase<T>(index) {}
};

//! Distinct type for edge index.
template <class T>
struct EdgeIndex : public FieldIndexBase<T> {
  EdgeIndex() :
    FieldIndexBase<T>(0) {}
  EdgeIndex(T index) :
    FieldIndexBase<T>(index) {}
};

//! Distinct type for face index.
template <class T>
struct FaceIndex : public FieldIndexBase<T> {
  FaceIndex() :
    FieldIndexBase<T>(0) {}
  FaceIndex(T index) :
    FieldIndexBase<T>(index) {}
};

//! Distinct type for cell index.
template <class T>
struct CellIndex : public FieldIndexBase<T> {
  CellIndex() :
    FieldIndexBase<T>(0) {}
  CellIndex(T index) :
    FieldIndexBase<T>(index) {}
};

// the following operators only exist to get the generic interpolate to compile
// 
template <class T>
vector<CellIndex<T> >
operator*(const vector<CellIndex<T> >& r, double &) {
  ASSERTFAIL("FieldIndex.h Bogus operator");
  return r;
}

template <class T>
vector<CellIndex<T> >
operator+=(const vector<CellIndex<T> >& l, const vector<CellIndex<T> >& r) {
  ASSERTFAIL("FieldIndex.h Bogus operator");
  return l;
}

template <class T>
const TypeDescription* get_type_description(NodeIndex<T>*)
{
  static TypeDescription* td = 0;

  if(!td){
    const TypeDescription *sub = SCIRun::get_type_description((T*)0);
    TypeDescription::td_vec *subs = scinew TypeDescription::td_vec(1);
    (*subs)[0] = sub;
    td = scinew TypeDescription("NodeIndex", subs, __FILE__, "SCIRun");
  }
  return td;
}

template <class T>
const TypeDescription* get_type_description(EdgeIndex<T>*)
{
  static TypeDescription* td = 0;
  if(!td){
    const TypeDescription *sub = SCIRun::get_type_description((T*)0);
    TypeDescription::td_vec *subs = scinew TypeDescription::td_vec(1);
    (*subs)[0] = sub;
    td = scinew TypeDescription("EdgeIndex", subs, __FILE__, "SCIRun");
  }
  return td;
}

template <class T>
const TypeDescription* get_type_description(FaceIndex<T>*)
{
  static TypeDescription* td = 0;
  if(!td){
    const TypeDescription *sub = SCIRun::get_type_description((T*)0);
    TypeDescription::td_vec *subs = scinew TypeDescription::td_vec(1);
    (*subs)[0] = sub;
    td = scinew TypeDescription("FaceIndex", subs, __FILE__, "SCIRun");
  }
  return td;
}

template <class T>
const TypeDescription* get_type_description(CellIndex<T>*)
{
  static TypeDescription* td = 0;
  if(!td){
    const TypeDescription *sub = SCIRun::get_type_description((T*)0);
    TypeDescription::td_vec *subs = scinew TypeDescription::td_vec(1);
    (*subs)[0] = sub;
    td = scinew TypeDescription("CellIndex", subs, __FILE__, "SCIRun");
  }
  return td;
}

#define FIELDINDEXBASE_VERSION 1

template<class T>
void Pio(Piostream& stream, FieldIndexBase<T>& data)
{
  Pio(stream, data.index_);
}

template <class T> const string find_type_name(NodeIndex<T> *)
{
  static const string name =
    string("NodeIndex") + FTNS + find_type_name((T *)0) + FTNE;
  return name;
}

template <class T> const string find_type_name(EdgeIndex<T> *)
{
  static const string name =
    string("EdgeIndex") + FTNS + find_type_name((T *)0) + FTNE;
  return name;
}

template <class T> const string find_type_name(FaceIndex<T> *)
{
  static const string name =
    string("FaceIndex") + FTNS + find_type_name((T *)0) + FTNE;
  return name;
}

template <class T> const string find_type_name(CellIndex<T> *)
{
  static const string name =
    string("CellIndex") + FTNS + find_type_name((T *)0) + FTNE;
  return name;
}

} // end namespace SCIRun

#endif // Datatypes_FieldIndex_h
