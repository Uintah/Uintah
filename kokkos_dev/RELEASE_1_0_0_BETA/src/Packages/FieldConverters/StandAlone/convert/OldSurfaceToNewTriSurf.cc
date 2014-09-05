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
 *  OldSurfaceToNewTriSurf.cc
 *
 *  Written by:
 *   David Weinstein and Michae Callahan
 *   Department of Computer Science
 *   University of Utah
 *   February 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Packages/FieldConverters/Core/Datatypes/TriSurface.h>
#include <Core/Datatypes/TriSurf.h>
#include <Core/Persistent/Pstreams.h>
#include <Core/Containers/HashTable.h>
#include <Core/Datatypes/FieldSet.h>
#include <Core/Geometry/Vector.h>

#include <iostream>
#include <fstream>
#include <stdlib.h>

using std::cerr;
using std::ifstream;
using std::endl;

using namespace SCIRun;
using namespace FieldConverters;

int
main(int argc, char **argv) {
  SurfaceHandle handle;
  
  // Read in a TriSurface
  if (argc !=3) {
    cerr << "Usage: "<<argv[0]<<" OldSurf NewTriSurf\n";
    exit(0);
  }
  Piostream* stream=auto_istream(argv[1]);
  if (!stream) {
    cerr << "Error - couldn't open file "<<argv[1]<<".  Exiting...\n";
    exit(0);
  }
  Pio(*stream, handle);
  if (!handle.get_rep()) {
    cerr << "Error reading Surface from file "<<argv[1]<<".  Exiting...\n";
    exit(0);
  }
  TriSurface *ts = dynamic_cast<TriSurface*>(handle.get_rep());
  if (!ts) {
    cerr << "Error - surface wasn't a TriSurface.\n";
  }

  
  TriSurfMesh *tsm = new TriSurfMesh();


  int i;
  for (i=0; i<ts->points.size(); i++)
  {
    tsm->add_point(ts->points[i]);
  }

  for (i=0; i<ts->elements.size(); i++)
  {
    const TSElement *ele = ts->elements[i];
    tsm->add_triangle(ele->i1, ele->i2, ele->i3);
  }

  tsm->connect();
  TriSurfMeshHandle tsmh(tsm);

  BinaryPiostream out_stream(argv[2], Piostream::Write);

#if 0
  // Package up the boundary conditions into a field.
  Field::data_location bcloc = Field::NONE;
  if (ts->valType == TriSurface::NodeType)
  {
    bcloc = Field::NODE;
  }
  else
  {
    bcloc = Field::FACE;
  }

  GenericField<TriSurfMesh, HashTable<int, double> > *bcfield =
    new GenericField<TriSurfMesh, HashTable<int, double> >(tsmh, bcloc);

  HashTable<int, double> &table = bcfield->fdata();
  
  for (i=0; i<ts->bcIdx.size(); i++)
  {
    table.insert(ts->bcIdx[i],ts->bcVal[i])
  }


  // Package up the normals into a field.
  Field::data_location loc = Field::NONE;
  switch (ts->normType)
  {
  case TriSurface::PointType:
    loc = Field::NODE;
    break;
  case TriSurface::VertexType:
    loc = Field::EDGE;
    break;
  case TriSurface::ElementType:
    loc = Field::FACE;
    break;
  default:
    loc = Field::NONE;
  }
  if (loc != Field::NONE)
  {
    TriSurf<Vector> *normals = new TriSurf<Vector>(tsmh, loc);
    vector<Vector> &vec = normals->fdata();
    for (int i = 0; i< ts->normals.size(); i++)
    {
      vec.push_back(ts->normals[i]);
    }

    FieldSet *fset = new FieldSet();
    //fset->add(GenericField<TriSurfMesh, HashTable<int, double> >::handle_type(bcfield));
    //fset->add(TriSurf<Vector>::handle_type(normals));

    FieldSetHandle fsetH(fset);

    // Write out the new field.
    Pio(out_stream, fsetH);
  }
  else
  {
    //GenericField<TriSurfMesh, HashTable<int, double> >::handle_type bch(bcfield);    
    //Pio(out_stream, bch);
  }

#endif
  return 0;  
}    
