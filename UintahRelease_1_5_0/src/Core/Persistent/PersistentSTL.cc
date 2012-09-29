/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

/*
 * PersistentSTL.cc: Persistent i/o for STL containers
 *    Author: Michael Callahan
 *            Department of Computer Science
 *            University of Utah
 *            March 2001
 * 
 */

#include <Core/Persistent/Persistent.h>
#include <Core/Persistent/PersistentSTL.h>

namespace SCIRun {

using std::vector;

template <> void Pio(Piostream& stream, vector<bool>& data)
{
  stream.begin_class("STLVector", STLVECTOR_VERSION);
  
  int size=(int)data.size();
  stream.io(size);
  
  if(stream.reading()){
    data.resize(size);
  }

  for (int i = 0; i < size; i++)
  {
    bool b;
    Pio(stream, b);
    data[i] = b;
  }

  stream.end_class();  
}



template <class T>
static inline void block_pio(Piostream &stream, vector<T> &data)
{
  if (stream.reading() && stream.peek_class() == "Array1")
  {
    stream.begin_class("Array1", STLVECTOR_VERSION);
  }
  else
  {
    stream.begin_class("STLVector", STLVECTOR_VERSION);
  }
  
  int size = (int)data.size();
  stream.io(size);
  
  if(stream.reading()){
    data.resize(size);
  }

  if (!stream.block_io(&data.front(), sizeof(T), data.size()))
  {
    for (int i = 0; i < size; i++)
    {
      Pio(stream, data[i]);
    }
  }

  stream.end_class();  
}


template <>
void Pio(Piostream& stream, vector<char>& data)
{
  block_pio(stream, data);
}

template <>
void Pio(Piostream& stream, vector<unsigned char>& data)
{
  block_pio(stream, data);
}

template <>
void Pio(Piostream& stream, vector<short>& data)
{
  block_pio(stream, data);
}

template <>
void Pio(Piostream& stream, vector<unsigned short>& data)
{
  block_pio(stream, data);
}

template <>
void Pio(Piostream& stream, vector<int>& data)
{
  block_pio(stream, data);
}

template <>
void Pio(Piostream& stream, vector<unsigned int>& data)
{
  block_pio(stream, data);
}

template <>
void Pio(Piostream& stream, vector<float>& data)
{
  block_pio(stream, data);
}

template <>
void Pio(Piostream& stream, vector<double>& data)
{
  block_pio(stream, data);
}


} // End namespace SCIRun
