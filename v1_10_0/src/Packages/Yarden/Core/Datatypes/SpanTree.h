
/*
 *  Span.h: The Span Data type
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   Nov. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */

#ifndef SCI_project_SpanTree_h
#define SCI_project_SpanTree_h 1

#include <Core/Datatypes/Datatype.h>
#include <Core/Containers/Array1.h>
#include <Core/Datatypes/ScalarField.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Geometry/BBox.h>

namespace Yarden {

using namespace SCIRun;

//#define FIELD_FLOAT

#ifdef FIELD_FLOAT
typedef float Value;
#else
typedef double Value;
#endif

struct SpanPoint {
  Value min;
  Value max;
  int index;
  
  SpanPoint(){}
  SpanPoint(Value _min, Value _max, int i) : min(_min), max(_max), index(i) {}
};

class SpanForest;
typedef LockingHandle<SpanForest> SpanForestHandle;

class SpanTree {
public:
  Array1<SpanPoint> span;
  
public:
  SpanTree(){}
  ~SpanTree() {}
};

class SpanForest : public Datatype {
public:
  Array1<SpanTree> tree;
  ScalarFieldHandle field;
  int generation;
  BBox bbox;
  int dx, dy;
  
public:
  SpanForest(){}
  virtual ~SpanForest() {}
  
  // Persistent representation
  virtual void io(Piostream&) {};
  static PersistentTypeID type_id;
};

} // End namespace SCIRun


#endif /* SCI_project_SpanTree_h */
