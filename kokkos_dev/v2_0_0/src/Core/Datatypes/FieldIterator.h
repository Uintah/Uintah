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
 *  FieldIterator.h: Some convenient simple iterators for fields.
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

#ifndef Datatypes_FieldIterator_h
#define Datatypes_FieldIterator_h

#include <Core/Datatypes/FieldIndex.h>

namespace SCIRun {


//! Base type for FieldIterator types.
template <class T>
struct FieldIteratorBase {
  FieldIteratorBase(T i) :
    index_(i) {}

  //! Field Iterators need to be able to increment.
  inline 
  T operator ++() { return ++index_; }
  
  bool operator ==(const FieldIteratorBase &a) const 
  { return index_ == a.index_; }
  bool operator !=(const FieldIteratorBase &a) const 
  { return index_ != a.index_; }

  inline T operator*() { return index_; }

protected:
  T index_;
#ifdef __digital__
public:
#else
private:
#endif
  //! Hide this in private to prevent it from being called.
  FieldIteratorBase<T> operator ++(int) { FieldIteratorBase<T> tmp(*this); ++index_; return tmp; }
};

//! Distinct type for node FieldIterator.
template <class T>
struct NodeIterator : public FieldIteratorBase<T> {
  NodeIterator() :
    FieldIteratorBase<T>(0) {}
  NodeIterator(T iter) :
    FieldIteratorBase<T>(iter) {}

  //! Required interface for an FieldIterator.
  inline 
  NodeIndex<T> operator*() { return NodeIndex<T>(index_); }
};

//! Distinct type for edge Iterator.
template <class T>
struct EdgeIterator : public FieldIteratorBase<T> {
  EdgeIterator() :
    FieldIteratorBase<T>(0) {}
  EdgeIterator(T index) :
    FieldIteratorBase<T>(index) {}

  //! Required interface for an FieldIterator.
  inline 
  EdgeIndex<T> operator*() { return EdgeIndex<T>(index_); }
};

//! Distinct type for face Iterator.
template <class T>
struct FaceIterator : public FieldIteratorBase<T> {
  FaceIterator() :
    FieldIteratorBase<T>(0) {}
  FaceIterator(T index) :
    FieldIteratorBase<T>(index) {}

  //! Required interface for an FieldIterator.
  inline 
  FaceIndex<T> operator*() { return FaceIndex<T>(index_); }
};

//! Distinct type for cell Iterator.
template <class T>
struct CellIterator : public FieldIteratorBase<T> {
  CellIterator() :
    FieldIteratorBase<T>(0) {}
  CellIterator(T index) :
    FieldIteratorBase<T>(index) {}

  //! Required interface for an FieldIterator.
  inline 
  CellIndex<T> operator*() { return CellIndex<T>(index_); }
};


}

#endif // Datatypes_FieldIterator_h
