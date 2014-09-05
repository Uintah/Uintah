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
 *  PointCloudToCVRTI.cc
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   May 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Core/Datatypes/PointCloud.h>
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
  if (handle->get_type_name(0) != "PointCloud") {
    cerr << "Error -- input field wasn't a PointCloud (type_name="<<handle->get_type_name(0)<<"\n";
    exit(0);
  }

  MeshHandle mb = handle->mesh();
  PointCloudMesh *tsm = dynamic_cast<PointCloudMesh *>(mb.get_rep());
  
  char fname[200];
  sprintf(fname, "%s.pts", argv[2]);
  FILE *fpts = fopen(fname, "wt");
  PointCloudMesh::Node::iterator niter; tsm->begin(niter);
  PointCloudMesh::Node::iterator niter_end; tsm->end(niter_end);
  PointCloudMesh::Node::size_type nsize; tsm->size(nsize);
  const unsigned int s = nsize;
  cerr << "Writing "<< s << " points to " << fname << "\n";
  while(niter != niter_end)
  {
    Point p;
    tsm->get_center(p, *niter);
    fprintf(fpts, "%lf %lf %lf\n", p.x(), p.y(), p.z());
    ++niter;
  }
  fclose(fpts);

  if (handle->get_type_name(1) == "Vector") {
    PointCloud<Vector> *fld = dynamic_cast<PointCloud<Vector> *>(handle.get_rep());
    sprintf(fname, "%s.grad", argv[2]);
    FILE *fgrad = fopen(fname, "wt");
    cerr << "Writing "<<fld->fdata().size()<<" vectors to "<<fname<<"\n";
    for (unsigned int i=0; i<fld->fdata().size(); i++) {
      Vector v = fld->fdata()[i];
      fprintf(fgrad, "%lf %lf %lf\n", v.x(), v.y(), v.z());
    }
  } else if (handle->get_type_name(1) == "double") {
    PointCloud<double> *fld = dynamic_cast<PointCloud<double> *>(handle.get_rep());
    sprintf(fname, "%s.pot", argv[2]);
    FILE *fpot = fopen(fname, "wt");
    cerr << "Writing "<<fld->fdata().size()<<" scalars to "<<fname<<"\n";
    for (unsigned int i=0; i<fld->fdata().size(); i++) {
      fprintf(fpot, "%lf\n", fld->fdata()[i]);
    }
  } else {
    cerr << "Unrecognized data template ("<<handle->get_type_name(1)<<") -- only mesh was output.\n";
  }
  return 0;  
}    
