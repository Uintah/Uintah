/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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


#include <Core/Datatypes/Mesh.h>
#include <Core/Geometry/Transform.h>
#include <Core/Geometry/BBox.h>

namespace SCIRun{

// initialize the static member type_id
PersistentTypeID Mesh::type_id("Mesh", "PropertyManager", NULL);

Mesh::Mesh() :
  MIN_ELEMENT_VAL(1.0e-12)
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
