
//----------------------------------------------------------------------
// PersistentMap.h: Persistent i/o for STL maps
// Author: David Hart
// Department of Computer Science
// University of Utah
// Mar. 2000
// Copyright (C) 2000 SCI Group
//----------------------------------------------------------------------

#ifndef SCI_project_Persistent_map_h
#define SCI_project_Persistent_map_h 1

#include <SCICore/Persistent/Persistent.h>

#define MAP_VERSION 1

namespace SCICore {
namespace PersistentSpace {

//----------------------------------------------------------------------

				// persistent io for maps
template <class Key, class Data>
SCICORESHARE inline void
  Pio(Piostream& stream, map<Key, Data, less<Key> >& data) {

#ifdef __GNUG__
  using namespace SCICore::PersistentSpace;
  using namespace SCICore::GeomSpace;
  using namespace SCICore::Containers;
  using namespace SCICore::Datatypes;
#else
  using SCICore::PersistentSpace::Pio;
  using SCICore::GeomSpace::Pio;
  using SCICore::Containers::Pio;
  using SCICore::Datatypes::Pio;
#endif

  map<Key, Data, less<Key> >::iterator iter;
  int i, n;
  Key k;
  Data d;
  
  stream.begin_class("Map", MAP_VERSION);

				// if reading from stream
  if (stream.reading()) {	
				// get map size 
    Pio(stream, n);
				// read elements
    for (i = 0; i < n; i++) {
      Pio(stream, k);
      Pio(stream, d);
      data[k] = d;
    }
    
  }
				// if writing to stream
  else {
				// write map size
    int n = data.size();
    Pio(stream, n);
				// write elements
    for (iter = data.begin(); iter != data.end(); iter++) {
				// have to copy iterator elements,
				// since passing them directly in a
				// call to Pio can be invalid because
				// Pio passes data by reference
      Key ik = (*iter).first;
      Data dk = (*iter).second;
      Pio(stream, ik);
      Pio(stream, dk);
    }
    
  }

  stream.end_class();
  
}


} // End namespace PersistentSpace
} // End namespace SCICore


#endif // SCI_project_Persistent_map_h
