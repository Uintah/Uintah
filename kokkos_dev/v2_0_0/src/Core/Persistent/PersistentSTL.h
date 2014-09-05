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
 * PersistentSTL.h: Persistent i/o for STL containers
 *    Author: David Hart, Alexei Samsonov
 *            Department of Computer Science
 *            University of Utah
 *            Mar. 2000, Dec 2000
 *    Copyright (C) 2000 SCI Group
 * 
 */

#ifndef SCI_project_PersistentSTL_h
#define SCI_project_PersistentSTL_h 1

#include <Core/Persistent/Persistent.h>
#include <sgi_stl_warnings_off.h>
#include <map>
#include <vector>
#include <list>
#include <sgi_stl_warnings_on.h>

namespace SCIRun {

using std::map;
using std::vector;
using std::list;
using std::pair;

#define MAP_VERSION 1


// Persistent IO for maps
template <class Key, class Data>
SCICORESHARE void
Pio(Piostream& stream, map<Key, Data>& data );

//////////
// Persistent IO of vector containers
template <class T>
SCICORESHARE void Pio(Piostream& stream, vector<T>& data);

template<class T,class S>
SCICORESHARE void 
Pio( Piostream &stream, pair<T,S>& pair);

//////////
// Persistent io for maps
template <class Key, class Data>
SCICORESHARE inline void
Pio(Piostream& stream, map<Key, Data>& data) {

#ifdef __GNUG__
#else
#endif

  typename map<Key, Data>::iterator iter;
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

//////////
// PIO for vectors
#define STLVECTOR_VERSION 2

template <> 
SCICORESHARE void Pio(Piostream& stream, vector<bool>& data);

template <class T> 
SCICORESHARE void Pio(Piostream& stream, vector<T>& data)
{ 
  if (stream.reading() && stream.peek_class() == "Array1")
  {
    stream.begin_class("Array1", STLVECTOR_VERSION);
  }
  else
  {
    stream.begin_class("STLVector", STLVECTOR_VERSION);
  }
  
  int size=(int)data.size();
  stream.io(size);
  
  if(stream.reading()){
    data.resize(size);
  }

  for (int i = 0; i < size; i++)
  {
    Pio(stream, data[i]);
  }

  stream.end_class();  
}


//////////
// PIO for lists
#define STLLIST_VERSION 1

template <class T> 
SCICORESHARE void Pio(Piostream& stream, list<T>& data)
{ 
  stream.begin_cheap_delim();
  
  int size=data.size();
  stream.io(size);
  
  if(stream.reading()){
    data.resize(size);
  }
  
  typename list<T>::iterator ii;
  for (ii=data.begin(); ii!=data.end(); ii++)
    Pio(stream, *ii);
     
  stream.end_cheap_delim();  
}


//////////
// PIO for pait
#define STLPAIR_VERSION 1

template <class T,class S> 
SCICORESHARE void Pio(Piostream& stream, pair<T,S>& data)
{ 
  
  stream.begin_class("STLPair", STLPAIR_VERSION);
  
  Pio(stream, data.first);
  Pio(stream, data.second);

  stream.end_class();  
}

} // End namespace SCIRun

#endif // SCI_project_PersistentSTL_h

