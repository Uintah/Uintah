
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

#include <SCICore/Datatypes/Datatype.h>
#include <SCICore/Containers/Array1.h>
#include <SCICore/Datatypes/ScalarField.h>
#include <SCICore/Containers/LockingHandle.h>
#include <SCICore/Geometry/BBox.h>

namespace PSECore {
namespace Datatypes {

using namespace SCICore::Datatypes;
 using namespace SCICore::Containers;
 using namespace SCICore::Geometry;

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

} // End namespace Datatypes
} // End namespace PSECore}


#endif /* SCI_project_SpanTree_h */
