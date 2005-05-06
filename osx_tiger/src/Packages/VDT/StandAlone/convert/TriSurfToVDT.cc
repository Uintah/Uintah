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
 *  TriSurfFieldToVDT.cc
 *      Builds a mesh between two SCIRun surfaces (and by default also fills
 *      in elements with the inner surface).
 *      Assumes that surfaces are like CVRTI format and standard SCIRun format
 *      in terms of surface normals (if they look good in the Viewer, with
 *      back-face culling turned on, they're good).  Inner surface can either
 *      be "hollow" (e.g. nothing inside heart), as specified by the optional
 *      -hollow command-line flag, or it can be filled (default).
 *      Sets up a .vin file for VDT mesh generation.  If -hollow is specified
 *      only one material region (between inner and outer surfaces) will be
 *      meshed with tets; otherwise two regions will be meshed (inner-most
 *      region will get material tag 2; region between inner and outer surface
 *      will get material tag 1).
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
  if (argc != 4 && argc != 5) {
    cerr << "Usage: "<<argv[0]<<" scirun_inner_trisurf scirun_outer_trisurf vdt_vin_file [-hollow]\n";
    return 0;
  }

  bool hollow=false;
  if (argc == 5)
    if (string(argv[4]) == "-hollow")
      hollow=true;

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
    cerr << "Error -- input field wasn't a TriSurfField (type_name="<<inner_surf->get_type_description(0)->get_name(0)<<"\n";
    exit(0);
  }
  TriSurfField<double> *inner_field =
    dynamic_cast<TriSurfField<double> *>(inner_surf.get_rep());

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

  FILE *fout = fopen(argv[3], "wt");
  if (!fout) {
    cerr << "Error - can't open input file "<<argv[3]<<"\n";
    exit(0);
  }

  TriSurfMesh::Node::iterator niter, niter_end;
  TriSurfMesh::Node::size_type ninner; inner->size(ninner);
  Point p0, p1;
  inner->begin(niter);
  inner->get_center(p0, *niter);
  ++niter;
  inner->get_center(p1, *niter);
  
  fprintf(fout, "input_version \"$Id$\"\n");
  fprintf(fout, "mesh_size %lf\n", (p0-p1).length());
  fprintf(fout, "gen_interior_pts\n");

  int i;
  Point p;
  inner->begin(niter);
  inner->end(niter_end);
  for (i=0; niter != niter_end; i++, ++niter) {
    inner->get_center(p, *niter);
    fprintf(fout, "pt %d %lf %lf %lf\n", i+1, p.x(), p.y(), p.z());
  }

  outer->begin(niter);
  outer->end(niter_end);
  for (; niter != niter_end; i++, ++niter) {
    outer->get_center(p, *niter);
    fprintf(fout, "pt %d %lf %lf %lf\n", i+1, p.x(), p.y(), p.z());
  }

  TriSurfMesh::Face::iterator fiter, fiter_end;
  TriSurfMesh::Node::array_type fac_nodes(3);
  fprintf(fout, "! boundary facets\n");
  inner->begin(fiter);
  inner->end(fiter_end);
  int maxval = 1;
  for (i=0; fiter != fiter_end; i++, ++fiter) {
    inner->get_nodes(fac_nodes, *fiter);
    int i1, i2, i3;
    i1=fac_nodes[1]+1;
    i2=fac_nodes[0]+1;
    i3=fac_nodes[2]+1;
    double val;
    inner_field->value(val, *fiter);
    int ival = (int)val;
    if (ival < 1) ival = 1;
    maxval = Max(ival, maxval);
    if (hollow)
      fprintf(fout, "bsf %d %d %d %d 1 0\n", i1, i2, i3, ival);
    else
      fprintf(fout, "bsf %d %d %d %d 1 2\n", i1, i2, i3, ival);
  }
  maxval++;
  outer->begin(fiter);
  outer->end(fiter_end);
  for (; fiter != fiter_end; i++, ++fiter) {
    outer->get_nodes(fac_nodes, *fiter);
    int i1, i2, i3;
    i1=fac_nodes[0]+1+ninner;
    i2=fac_nodes[1]+1+ninner;
    i3=fac_nodes[2]+1+ninner;
    fprintf(fout, "bsf %d %d %d %d 1 0\n", i1, i2, i3, maxval);
  }
  fclose(fout);
  return 0;
}    
