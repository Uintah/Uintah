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
 *  TriSurfToVtk.cc
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   May 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Core/Datatypes/TriSurfField.h>
#include <Core/Persistent/Pstreams.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>

using std::cerr;
using std::ifstream;
using std::endl;

using namespace SCIRun;

int
main(int argc, char **argv) {
  if (argc != 3) {
    cerr << "Usage: "<<argv[0]<<" scirun_field vtk_poly\n";
    return 0;
  }

  FieldHandle handle;
  Piostream* stream=auto_istream(argv[1]);
  if (!stream) {
    cerr << "Couldn't open file "<<argv[1]<<".  Exiting...\n";
    exit(0);
  }
  Pio(*stream, handle);
  if (!handle.get_rep()) {
    cerr << "Error reading surface from file "<<argv[1]<<".  Exiting...\n";
    exit(0);
  }
  if (handle->get_type_name(0) != "TriSurfField") {
    cerr << "Error -- input field wasn't a TriSurfField (type_name="<<handle->get_type_name(0)<<"\n";
    exit(0);
  }

  MeshHandle mb = handle->mesh();
  TriSurfMesh *tsm = dynamic_cast<TriSurfMesh *>(mb.get_rep());
  FILE *fout = fopen(argv[2], "wt");
  if (!fout) {
    cerr << "Error - can't open input file "<<argv[2]<<"\n";
    exit(0);
  }

  fprintf(fout, "# vtk DataFile Version 3.0\nvtk output\nASCII\nDATASET POLYDATA\n");
  TriSurfMesh::Node::iterator niter; tsm->begin(niter);
  TriSurfMesh::Node::iterator niter_end; tsm->end(niter_end);
  TriSurfMesh::Node::size_type nsize; tsm->size(nsize);
  cerr << "Writing "<<((unsigned int)nsize)<< " points to "<<argv[2]<<"...\n";
  fprintf(fout, "POINTS %d float\n", (int)nsize);
  while(niter != niter_end)
  {
    Point p;
    tsm->get_center(p, *niter);
    fprintf(fout, "%lf %lf %lf\n", p.x(), p.y(), p.z());
    ++niter;
  }

  TriSurfMesh::Face::size_type fsize; tsm->size(fsize);
  TriSurfMesh::Face::iterator fiter; tsm->begin(fiter);
  TriSurfMesh::Face::iterator fiter_end; tsm->end(fiter_end);
  TriSurfMesh::Node::array_type fac_nodes(3);
  cerr << "     and "<< ((unsigned int)fsize)<<" faces.\n";
  fprintf(fout, "POLYGONS %d %d\n", (int)fsize, (int)fsize*4);
  while(fiter != fiter_end)
  {
    tsm->get_nodes(fac_nodes, *fiter);
    fprintf(fout, "3 %d %d %d\n", (int)fac_nodes[0], (int)fac_nodes[1], (int)fac_nodes[2]);
    ++fiter;
  }
  fclose(fout);
  return 0;  
}    
