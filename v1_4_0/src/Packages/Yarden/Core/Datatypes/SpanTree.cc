/*
 *  SpanTree.cc: The SpanTree Data type
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   Nov. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */

#include <Packages/Yarden/Core/Datatypes/SpanTree.h>

namespace Yarden {

using namespace SCIRun;

PersistentTypeID SpanForest::type_id("SpanTree", "Datatype", 0);

#define SpanForest_VERSION 0

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
