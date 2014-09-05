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
 *  PCA-example.cc
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 2003
 *
 *  Copyright (C) 2003 SCI Group
 */

// This program will read in a SCIRun PointCloudField, builds covariance
// matrices from the points, performs PCA on the matrix, and prints out
// the eigenvalues / eigenvectors.

#include <Core/Datatypes/PointCloudField.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Persistent/Pstreams.h>
#include <Core/Containers/HashTable.h>
#include <StandAlone/convert/FileUtils.h>
#if defined(__APPLE__)
#  include <Core/Datatypes/MacForceLoad.h>
#endif
#include <iostream>
#include <fstream>
#include <stdlib.h>

using std::cerr;
using std::ifstream;
using std::endl;

using namespace SCIRun;

int
main(int argc, char **argv) {
  if (argc != 2) {
    cerr << "Usage: "<<argv[0]<<" PointCloudField\n";
    return 0;
  }
#if defined(__APPLE__)  
  macForceLoad(); // Attempting to force load (and thus instantiation of
	          // static constructors) Core/Datatypes;
#endif
  char *fieldName = argv[1];

  FieldHandle handle;
  Piostream* stream=auto_istream(fieldName);
  if (!stream) {
    cerr << "Couldn't open file "<<fieldName<<".  Exiting...\n";
    exit(0);
  }
  Pio(*stream, handle);
  if (!handle.get_rep()) {
    cerr << "Error reading surface from file "<<fieldName<<".  Exiting...\n";
    exit(0);
  }
  if (handle->get_type_description(0)->get_name() != "PointCloudField") {
    cerr << "Error -- input field wasn't a PointCloudField (type_name="<<handle->get_type_description(0)->get_name()<<"\n";
    exit(0);
  }

  MeshHandle mH = handle->mesh();
  PointCloudMesh *pcm = dynamic_cast<PointCloudMesh *>(mH.get_rep());
  PointCloudMesh::Node::iterator niter; 
  PointCloudMesh::Node::iterator niter_end; 
  PointCloudMesh::Node::size_type nsize; 
  pcm->size(nsize);
  cerr << "\n\nNumber of points = "<< nsize <<"\n";
  pcm->begin(niter);
  pcm->end(niter_end);
  Vector v;
  while(niter != niter_end) {
    Point p;
    pcm->get_center(p, *niter);
    v+=p.vector();
    ++niter;
  }
  v/=nsize;
  pcm->begin(niter);
  pcm->end(niter_end);
  DenseMatrix dm(3,3);
  DenseMatrix dmTemp(3,3);
  while(niter != niter_end) {
    Point p;
    pcm->get_center(p, *niter);
    Point diff(p-v);
    dmTemp[0][0]=diff.x()*diff.x();
    dmTemp[1][1]=diff.y()*diff.y();
    dmTemp[2][2]=diff.z()*diff.z();
    dmTemp[0][1]=dmTemp[1][0]=diff.x()*diff.y();
    dmTemp[0][2]=dmTemp[2][0]=diff.x()*diff.z();
    dmTemp[1][2]=dmTemp[2][1]=diff.y()*diff.z();
    for (int ii=0; ii<3; ii++)
      for (int jj=0; jj<3; jj++)
	dm[ii][jj]+=dmTemp[ii][jj]/nsize;
    ++niter;
  }
  cerr << "Centroid="<<v<<"\n";
  ColumnMatrix R(3), I(3);
  DenseMatrix EVecs(3,3);
  dm.eigenvectors(R, I, EVecs);
  cerr << "\n  *** Real Evals ***\n";
  R.print();
  cerr << "\n  *** Imag Evals ***\n";
  I.print();
  cerr << "\n  *** Evecs ***\n";
  EVecs.print();
  cerr << "\n  *** Covar Mat ***\n";
  dm.print();
  cerr << "\n\n";
  return 0;  
}
