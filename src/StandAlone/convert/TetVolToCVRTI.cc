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
 *  TetVolFieldToCVRTI.cc
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   May 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Core/Datatypes/TetVolField.h>
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
  if (handle->get_type_description(0)->get_name() != "TetVolField") {
    cerr << "Error -- input field wasn't a TetVolField (type_name="<<handle->get_type_description(0)->get_name()<<"\n";
    exit(0);
  }

  MeshHandle mb = handle->mesh();
  TetVolMesh *tvm = dynamic_cast<TetVolMesh *>(mb.get_rep());
  
  char fname[200];
  sprintf(fname, "%s.pts", argv[2]);
  FILE *fpts = fopen(fname, "wt");
  TetVolMesh::Node::iterator niter; tvm->begin(niter);
  TetVolMesh::Node::iterator niter_end; tvm->end(niter_end);
  TetVolMesh::Node::size_type nsize; tvm->size(nsize);
  cerr << "Writing "<<((unsigned int)nsize)<<" points to "<<fname<<"\n";
  while(niter != niter_end)
  {
    Point p;
    tvm->get_center(p, *niter);
    fprintf(fpts, "%lf %lf %lf\n", p.x(), p.y(), p.z());
    ++niter;
  }
  fclose(fpts);

  sprintf(fname, "%s.tet", argv[2]);
  FILE *ftet = fopen(fname, "wt");
  TetVolMesh::Face::size_type fsize; tvm->size(fsize);
  cerr << "Writing "<<((unsigned int)fsize)<<" tets to "<<fname<<"\n";
  TetVolMesh::Cell::iterator citer; tvm->begin(citer);
  TetVolMesh::Cell::iterator citer_end; tvm->end(citer_end);
  TetVolMesh::Node::array_type cell_nodes(4);

  while(citer != citer_end)
  {
    tvm->get_nodes(cell_nodes, *citer);
    fprintf(ftet, "%d %d %d %d\n", cell_nodes[0]+1, cell_nodes[1]+1, cell_nodes[2]+1, cell_nodes[3]+1);
    ++citer;
  }

  if (handle->get_type_description(1)->get_name() == "Vector") {
    TetVolField<Vector> *fld = dynamic_cast<TetVolField<Vector> *>(handle.get_rep());
    sprintf(fname, "%s.grad", argv[2]);
    FILE *fgrad = fopen(fname, "wt");
    cerr << "Writing "<<fld->fdata().size()<<" vectors to "<<fname<<"\n";
    for (unsigned int i=0; i<fld->fdata().size(); i++) {
      Vector v = fld->fdata()[i];
      fprintf(fgrad, "%lf %lf %lf\n", v.x(), v.y(), v.z());
    }
  } else if (handle->get_type_description(1)->get_name() == "double") {
    TetVolField<double> *fld = dynamic_cast<TetVolField<double> *>(handle.get_rep());
    sprintf(fname, "%s.pot", argv[2]);
    FILE *fpot = fopen(fname, "wt");
    cerr << "Writing "<<fld->fdata().size()<<" scalars to "<<fname<<"\n";
    for (unsigned int i=0; i<fld->fdata().size(); i++) {
      fprintf(fpot, "%lf\n", fld->fdata()[i]);
    }
  } else {
    cerr << "Unrecognized data template ("<<handle->get_type_description(1)->get_name()<<") -- only mesh was output.\n";
  }
  return 0;  
}    
