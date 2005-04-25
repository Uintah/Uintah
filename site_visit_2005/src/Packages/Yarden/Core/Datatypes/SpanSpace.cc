/*
 *  SpanSpace.cc: The SpanSpace Data type
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   Nov. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */

#include <Core/Datatypes/Datatype.h>
#include <Core/Containers/Array1.h>
#include <Core/Datatypes/ScalarField.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Geometry/BBox.h>
#include <Packages/Yarden/Core/Datatypes/SpanSpace.h>


namespace Yarden {
    
using namespace SCIRun;

    SpanSpaceBuildUG::SpanSpaceBuildUG ( ScalarFieldUG *field)
      {
	Array1<double> &data = field->data;

	for (int i=0; i<field->mesh->elemsize(); i++ ) {
	  int *n = field->mesh->element(i)->n;
	  
	  double min, max;
	  min = max = data[n[0]];
	  for ( int j=1; j<4; j++) {
	    double v = data[n[j]];
	    if ( v < min ) min = v;
	    else if ( v > max ) max = v;
	  }
	  
	  span.add( SpanPoint<double>( min, max, i ) );
	}
      }

    PersistentTypeID SpanUniverse::type_id("SpanUniverse", "Datatype", 0);
    
#define SpanUniverse_VERSION 0
    
/*
void SpanForest::io(Piostream& stream) {
   int version=stream.begin_class("SpanForest", SpanForest_VERSION);
    Pio(stream, name);
    if (version >= 2) {
	Pio(stream, conductivity);
	int bt=bdry_type;
	Pio(stream, bt);
	if(stream.reading())
	    bdry_type=(Boundary_type)bt;
    }
    stream.end_class();
}
*/

    
} // End namespace Yarden
