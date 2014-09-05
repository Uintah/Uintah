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


#include <Core/Datatypes/MeshBase.h>

namespace SCIRun{

// initialize the static member type_id
PersistentTypeID MeshBase::type_id(type_name(), "MeshBaseData", NULL);


MeshBase::~MeshBase() 
{
}


const int MESHBASE_VERSION = 1;

void 
MeshBase::io(Piostream& stream) {

  stream.begin_class(type_name(-1), MESHBASE_VERSION);
  PropertyManager::io(stream);
  stream.end_class();
}

const string 
MeshBase::type_name(int n)
{
  ASSERT(n >= -1 && n <= 0);
  static const string name = "MeshBase";
  return name;
}


}
