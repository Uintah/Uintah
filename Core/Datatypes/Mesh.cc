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


#include <Core/Datatypes/Mesh.h>

namespace SCIRun{

// initialize the static member type_id
PersistentTypeID Mesh::type_id("Mesh", "PropertyManager", NULL);


Mesh::~Mesh() 
{
}


const int MESHBASE_VERSION = 2;

void 
Mesh::io(Piostream& stream)
{
  if (stream.reading() && stream.peek_class() == "MeshBase")
  {
    stream.begin_class("MeshBase", 1);
  }
  else
  {
    stream.begin_class("Mesh", MESHBASE_VERSION);
  }
  PropertyManager::io(stream);
  stream.end_class();
}

const string 
Mesh::type_name(int n)
{
  ASSERT(n >= -1 && n <= 0);
  static const string name = "Mesh";
  return name;
}


}
