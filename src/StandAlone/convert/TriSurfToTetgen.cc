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
 *  TriSurfFieldToVtk.cc
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
  if (argc != 4) {
    cerr << "Usage: "<<argv[0]<<" scirun_inner_trisurf scirun_outer_trisurf tetgen_basename\n";
    return 0;
  }

  FieldHandle inner_surf;
  Piostream* stream1=auto_istream(argv[1]);
  if (!stream1) {
    cerr << "Couldn't open file "<<argv[1]<<".  Exiting...\n";
    exit(0);
  }
  Pio(*stream1, inner_surf);
  if (!inner_surf.get_rep()) {
    cerr << "Error reading surface from file "<<argv[1]<<".  Exiting...\n";
    exit(0);
  }
  if (inner_surf->get_type_description(0)->get_name() != "TriSurfField") {
    cerr << "Error -- input field wasn't a TriSurfField (type_name="<<inner_surf->get_type_description(0)->get_name()<<"\n";
    exit(0);
  }

  MeshHandle mh = inner_surf->mesh();
  TriSurfMesh *inner = dynamic_cast<TriSurfMesh *>(mh.get_rep());

  FieldHandle outer_surf;
  Piostream* stream2=auto_istream(argv[2]);
  if (!stream2) {
    cerr << "Couldn't open file "<<argv[2]<<".  Exiting...\n";
    exit(0);
  }
  Pio(*stream2, outer_surf);
  if (!outer_surf.get_rep()) {
    cerr << "Error reading surface from file "<<argv[2]<<".  Exiting...\n";
    exit(0);
  }
  if (outer_surf->get_type_description(0)->get_name() != "TriSurfField") {
    cerr << "Error -- input field wasn't a TriSurfField (type_name="<<outer_surf->get_type_description(0)->get_name()<<"\n";
    exit(0);
  }
  mh = outer_surf->mesh();
  TriSurfMesh *outer = dynamic_cast<TriSurfMesh *>(mh.get_rep());

  char filename[1000];
  sprintf(filename, "%s.poly", argv[3]);
  FILE *fout = fopen(filename, "wt");
  if (!fout) {
    cerr << "Error - can't open input file "<<filename<<"\n";
    exit(0);
  }

  TriSurfMesh::Node::iterator niter, niter_end;
  TriSurfMesh::Node::size_type ninner; inner->size(ninner);
  TriSurfMesh::Face::size_type finner; inner->size(finner);
  TriSurfMesh::Node::size_type nouter; outer->size(nouter);
  TriSurfMesh::Face::size_type fouter; outer->size(fouter);
  
  fprintf(fout, "# Number of nodes, pts/tri, no holes, no boundary markers\n");
  fprintf(fout, "%d 3 0 0\n", ninner+nouter);
  fprintf(fout, "\n# Inner surface points\n");

  int i;
  Point p;
  inner->begin(niter);
  inner->end(niter_end);
  Point mid_inner;
  for (i=0; niter != niter_end; i++, ++niter) {
    inner->get_center(p, *niter);
    mid_inner += p.asVector();
    fprintf(fout, "%d %lf %lf %lf\n", i+1, p.x(), p.y(), p.z());
  }
  mid_inner /= ninner;

  fprintf(fout, "\n# Outer surface points\n");
  outer->begin(niter);
  outer->end(niter_end);
  for (; niter != niter_end; i++, ++niter) {
    outer->get_center(p, *niter);
    fprintf(fout, "%d %lf %lf %lf\n", i+1, p.x(), p.y(), p.z());
  }

  fprintf(fout, "\n# Number of faces, no boundary markers\n");
  fprintf(fout, "%d 0\n\n", finner+fouter);
  Point mid_outer = AffineCombination(mid_inner, 0.5, p, 0.5);

  TriSurfMesh::Face::iterator fiter, fiter_end;
  TriSurfMesh::Node::array_type fac_nodes(3);
  inner->begin(fiter);
  inner->end(fiter_end);
  fprintf(fout, "# Inner faces\n");
  for (i=0; fiter != fiter_end; i++, ++fiter) {
    inner->get_nodes(fac_nodes, *fiter);
    int i1, i2, i3;
    i1=fac_nodes[0]+1;
    i2=fac_nodes[1]+1;
    i3=fac_nodes[2]+1;
    fprintf(fout, "1\n3 %d %d %d\n", i1, i2, i3);
  }
  outer->begin(fiter);
  outer->end(fiter_end);
  fprintf(fout, "\n# Outer faces\n");
  for (; fiter != fiter_end; i++, ++fiter) {
    outer->get_nodes(fac_nodes, *fiter);
    int i1, i2, i3;
    i1=fac_nodes[0]+1+ninner;
    i2=fac_nodes[1]+1+ninner;
    i3=fac_nodes[2]+1+ninner;
    fprintf(fout, "1\n3 %d %d %d\n", i1, i2, i3);
  }
  fprintf(fout, "\n# No holes\n0\n\n# Two regions\n2\n\n");
  fprintf(fout, "1 %lf %lf %lf 1\n", mid_inner.x(), mid_inner.y(), mid_inner.z());
  fprintf(fout, "2 %lf %lf %lf 2\n", mid_outer.x(), mid_outer.y(), mid_outer.z());
  
  fclose(fout);
  return 0;
}    
