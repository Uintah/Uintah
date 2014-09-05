/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
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
#include <Core/Datatypes/GenericField.h>
#include <Core/Basis/TriLinearLgn.h>
#include <Core/Datatypes/TriSurfMesh.h>
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
  if (inner_surf->get_type_description(Field::FIELD_NAME_ONLY_E)->get_name() != "TriSurfField") {
    cerr << "Error -- input field wasn't a TriSurfField (type_name="<<inner_surf->get_type_description(Field::FIELD_NAME_ONLY_E)->get_name()<<"\n";
    exit(0);
  }
  typedef TriSurfMesh<TriLinearLgn<Point> > TSMesh;
  MeshHandle mh = inner_surf->mesh();
  TSMesh *inner = dynamic_cast<TSMesh *>(mh.get_rep());

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
  if (outer_surf->get_type_description(Field::FIELD_NAME_ONLY_E)->get_name() != "TriSurfField") {
    cerr << "Error -- input field wasn't a TriSurfField (type_name="<<outer_surf->get_type_description(Field::FIELD_NAME_ONLY_E)->get_name()<<"\n";
    exit(0);
  }
  mh = outer_surf->mesh();
  TSMesh *outer = dynamic_cast<TSMesh *>(mh.get_rep());

  char filename[1000];
  sprintf(filename, "%s.poly", argv[3]);
  FILE *fout = fopen(filename, "wt");
  if (!fout) {
    cerr << "Error - can't open input file "<<filename<<"\n";
    exit(0);
  }

  TSMesh::Node::iterator niter, niter_end;
  TSMesh::Node::size_type ninner; inner->size(ninner);
  TSMesh::Face::size_type finner; inner->size(finner);
  TSMesh::Node::size_type nouter; outer->size(nouter);
  TSMesh::Face::size_type fouter; outer->size(fouter);
  
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

  TSMesh::Face::iterator fiter, fiter_end;
  TSMesh::Node::array_type fac_nodes(3);
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
