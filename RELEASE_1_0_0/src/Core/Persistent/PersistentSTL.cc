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
 * PersistentSTL.cc: Persistent i/o for STL containers
 *    Author: Michael Callahan
 *            Department of Computer Science
 *            University of Utah
 *            March 2001
 *    Copyright (C) 2001 SCI Group
 * 
 */

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

} // End namespace SCIRun
