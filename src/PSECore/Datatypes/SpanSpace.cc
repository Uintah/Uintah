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

namespace PSECore {
  namespace Datatypes {
    
#ifdef UNIVERSE
    
    PersistentTypeID SpanUniverse::type_id("SpanUniverse", "Datatype", 0);
    
#define SpanUniverse_VERSION 0
#endif
    
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

    
  }
}
