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
 *  TriSurfToOBJ.cc
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
  if (argc != 7) {
    cerr << "Usage: "<<argv[0]<<" scirun_field output_basename color_r color_g color_b Ns\n";
    return 0;
  }

  double red=atof(argv[3]);
  double green=atof(argv[4]);
  double blue=atof(argv[5]);
  double Ns=atof(argv[6]);

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
  char objname[512];
  sprintf(objname, "%s.obj", argv[2]);
  char mtlname[512];
  sprintf(mtlname, "%s.mtl", argv[2]);
  FILE *fobj = fopen(objname, "wt");
  FILE *fmtl = fopen(mtlname, "wt");
  if (!fobj || !fmtl) {
    cerr << "Error - can't open output file "<<objname<<" or "<<mtlname<<"\n";
    exit(0);
  }

  tsm->synchronize(Mesh::NORMALS_E);
  TriSurfMesh::Node::iterator niter; 
  TriSurfMesh::Node::iterator niter_end; 
  TriSurfMesh::Node::size_type nsize; 
  tsm->begin(niter);
  tsm->end(niter_end);
  tsm->size(nsize);
  while(niter != niter_end)
  {
    Point p;
    tsm->get_center(p, *niter);
    fprintf(fobj, "v %lf %lf %lf\n", p.x(), p.y(), p.z());
    Vector n;
    tsm->get_normal(n, *niter);
    fprintf(fobj, "vn %lf %lf %lf\n", n.x(), n.y(), n.z());
    ++niter;
  }
  fprintf(fobj, "usemtl Default\n");  
  TriSurfMesh::Face::size_type fsize; 
  TriSurfMesh::Face::iterator fiter; 
  TriSurfMesh::Face::iterator fiter_end; 
  TriSurfMesh::Node::array_type fac_nodes(3);
  tsm->size(fsize);
  tsm->begin(fiter);
  tsm->end(fiter_end);
  while(fiter != fiter_end)
  {
    tsm->get_nodes(fac_nodes, *fiter);
    fprintf(fobj, "f  %d//%d %d//%d %d//%d\n", 
	    (int)fac_nodes[0]+1, (int)fac_nodes[0]+1,
	    (int)fac_nodes[1]+1, (int)fac_nodes[1]+1,
	    (int)fac_nodes[2]+1, (int)fac_nodes[2]+1);
	    
    ++fiter;
  }
  fclose(fobj);
  fprintf(fmtl,"newmtl Default\n");
  fprintf(fmtl," \tKa 0 0 0\n");
  fprintf(fmtl," \tKd %lf %lf %lf\n", red, green, blue);
  fprintf(fmtl," \tKs 1 1 1\n");
  fprintf(fmtl," \tillum 2\n");
  fprintf(fmtl," \tNs %lf\n", Ns);
  fclose(fmtl);
  return 0;  
}    
