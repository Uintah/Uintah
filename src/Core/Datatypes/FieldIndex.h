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

namespace SCIRun {


//! Base type for index types.
template <class T>
struct FieldIndexBase {
  typedef T value_type;
  
  FieldIndexBase(T i) :
    index_(i) {}

  //! Required interface for an Index.
  inline 
  T& operator*() { return index_; }

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


}

#endif // Datatypes_FieldIndex_h
