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
#include <Core/Geometry/Transform.h>
#include <Core/Geometry/BBox.h>

namespace SCIRun{

// initialize the static member type_id
PersistentTypeID Mesh::type_id("Mesh", "PropertyManager", NULL);

Mesh::Mesh()
{
}

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

const string
Mesh::get_type_name(int n) const
{
  ASSERT(n==0);
  return get_type_description()->get_name();
}


//! Return the transformation that takes a 0-1 space bounding box 
//! to the current bounding box of this mesh.
void Mesh::get_canonical_transform(Transform &t) 
{
  t.load_identity();
  BBox bbox = get_bounding_box();
  t.pre_scale(bbox.diagonal());
  t.pre_translate(Vector(bbox.min()));
}

}
