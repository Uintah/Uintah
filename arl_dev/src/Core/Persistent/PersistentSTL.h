/*
 * The MIT License
 *
 * Copyright (c) 1997-2015 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

/*
 * PersistentSTL.h: Persistent i/o for STL containers
 *    Author: David Hart, Alexei Samsonov
 *            Department of Computer Science
 *            University of Utah
 *            Mar. 2000, Dec 2000
 * 
 */

#ifndef SCI_project_PersistentSTL_h
#define SCI_project_PersistentSTL_h 1

#include <Core/Persistent/Persistent.h>

#include <map>
#include <vector>
#include <list>

namespace SCIRun {

#define MAP_VERSION 1


// Persistent IO for maps
template <class Key, class Data>
void
Pio(Piostream& stream, std::map<Key, Data>& data );

//////////
// Persistent IO of vector containers
template <class T>
void Pio(Piostream& stream, std::vector<T>& data);

template<class T,class S>
void 
Pio( Piostream &stream, std::pair<T,S>& pair);

//////////
// Persistent io for maps
template <class Key, class Data>
inline void
Pio(Piostream& stream, std::map<Key, Data>& data) {

#ifdef __GNUG__
#else
#endif

  typename std::map<Key, Data>::iterator iter;
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
void Pio(Piostream& stream, std::vector<bool>& data);


// Optimize these four, heavily used in the field classes.
template <>
void Pio(Piostream& stream, std::vector<char>& data);
template <>
void Pio(Piostream& stream, std::vector<unsigned char>& data);
template <>
void Pio(Piostream& stream, std::vector<short>& data);
template <>
void Pio(Piostream& stream, std::vector<unsigned short>& data);
template <>
void Pio(Piostream& stream, std::vector<int>& data);
template <>
void Pio(Piostream& stream, std::vector<unsigned int>& data);
template <>
void Pio(Piostream& stream, std::vector<float>& data);
template <>
void Pio(Piostream& stream, std::vector<double>& data);

template <class T> 
void Pio(Piostream& stream, std::vector<T>& data)
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
void Pio(Piostream& stream, std::list<T>& data)
{ 
  stream.begin_cheap_delim();
  
  int size=data.size();
  stream.io(size);
  
  if(stream.reading()){
    data.resize(size);
  }
  
  typename std::list<T>::iterator ii;
  for (ii=data.begin(); ii!=data.end(); ii++)
    Pio(stream, *ii);
     
  stream.end_cheap_delim();  
}


//////////
// PIO for pait
#define STLPAIR_VERSION 1

template <class T,class S> 
void Pio(Piostream& stream, std::pair<T,S>& data)
{ 
  
  stream.begin_class("STLPair", STLPAIR_VERSION);
  
  Pio(stream, data.first);
  Pio(stream, data.second);

  stream.end_class();  
}

} // End namespace SCIRun

#endif // SCI_project_PersistentSTL_h

