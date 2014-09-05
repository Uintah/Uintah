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
 *  TetVolToCVRTI.cc
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   May 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Core/Datatypes/TetVol.h>
#include <Core/Geometry/Vector.h>
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
    cerr << "Usage: "<<argv[0]<<" scirun_field output_base\n";
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
  if (handle->get_type_name(0) != "TetVol") {
    cerr << "Error -- input field wasn't a TetVol (type_name="<<handle->get_type_name(0)<<"\n";
    exit(0);
  }

  MeshBaseHandle mb = handle->mesh();
  TetVolMesh *tvm = dynamic_cast<TetVolMesh *>(mb.get_rep());
  
  char fname[200];
  sprintf(fname, "%s.pts", argv[2]);
  FILE *fpts = fopen(fname, "wt");
  TetVolMesh::Node::iterator niter = tvm->node_begin();
  cerr << "Writing "<<tvm->nodes_size()<<" points to "<<fname<<"\n";
  while(niter != tvm->node_end()) {
    Point p;
    tvm->get_center(p, *niter);
    fprintf(fpts, "%lf %lf %lf\n", p.x(), p.y(), p.z());
    ++niter;
  }
  fclose(fpts);

  sprintf(fname, "%s.tet", argv[2]);
  FILE *ftet = fopen(fname, "wt");
  cerr << "Writing "<<tvm->faces_size()<<" tets to "<<fname<<"\n";
  TetVolMesh::Cell::iterator citer = tvm->cell_begin();
  TetVolMesh::Node::array_type cell_nodes(4);

  while(citer != tvm->cell_end()) {
    tvm->get_nodes(cell_nodes, *citer);
    fprintf(ftet, "%d %d %d %d\n", cell_nodes[0]+1, cell_nodes[1]+1, cell_nodes[2]+1, cell_nodes[3]+1);
    ++citer;
  }

  if (handle->get_type_name(1) == "Vector") {
    TetVol<Vector> *fld = dynamic_cast<TetVol<Vector> *>(handle.get_rep());
    sprintf(fname, "%s.grad", argv[2]);
    FILE *fgrad = fopen(fname, "wt");
    cerr << "Writing "<<fld->fdata().size()<<" vectors to "<<fname<<"\n";
    for (int i=0; i<fld->fdata().size(); i++) {
      Vector v = fld->fdata()[i];
      fprintf(fgrad, "%lf %lf %lf\n", v.x(), v.y(), v.z());
    }
  } else if (handle->get_type_name(1) == "double") {
    TetVol<double> *fld = dynamic_cast<TetVol<double> *>(handle.get_rep());
    sprintf(fname, "%s.pot", argv[2]);
    FILE *fpot = fopen(fname, "wt");
    cerr << "Writing "<<fld->fdata().size()<<" scalars to "<<fname<<"\n";
    for (int i=0; i<fld->fdata().size(); i++) {
      fprintf(fpot, "%lf\n", fld->fdata()[i]);
    }
  } else {
    cerr << "Unrecognized data template ("<<handle->get_type_name(1)<<") -- only mesh was output.\n";
  }
  return 0;  
}    
