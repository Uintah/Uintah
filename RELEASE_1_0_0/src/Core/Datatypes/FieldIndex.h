/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
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
 *  Copyright (C) 2001 SCI Group
 *
 */

#ifndef Datatypes_FieldIndex_h
#define Datatypes_FieldIndex_h

#include <vector>
#include <Core/Persistent/Persistent.h>

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

#define FIELDINDEXBASE_VERSION 1

template<class T>
void Pio(Piostream& stream, FieldIndexBase<T>& data)
{
  Pio(stream, data.index_);
}

}

#endif // Datatypes_FieldIndex_h
