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
 *  TransformField.cc:  Rotate and flip field to get it into "standard" view
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   March 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Core/Containers/String.h>
#include <Dataflow/Network/Module.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Geometry/Transform.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/MiscMath.h>
#include <Core/Datatypes/TetVol.h>
#include <Core/Datatypes/LatticeVol.h>
#include <Core/Datatypes/TriSurf.h>
#include <Core/Datatypes/ContourField.h>
#include <Core/Datatypes/PointCloud.h>
#include <iostream>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

using std::cerr;

namespace SCIRun {


class TransformField : public Module
{
public:
  TransformField(const clString& id);
  virtual ~TransformField();

  virtual void execute();


  void matrix_to_transform(MatrixHandle mH, Transform& t);
  template <class M> void callback(Field *ifield, M *);
private:  
  Transform trans_;
};


extern "C" Module* make_TransformField(const clString& id) {
  return new TransformField(id);
}


TransformField::TransformField(const clString& id)
  : Module("TransformField", id, Source, "Fields", "SCIRun")
{
}


TransformField::~TransformField()
{
}


void
TransformField::matrix_to_transform(MatrixHandle mH, Transform& t)
{
  double a[16];
  double *p=&(a[0]);
  for (int i=0; i<4; i++)
    for (int j=0; j<4; j++)
      *p++=(*mH.get_rep())[i][j];
  t.set(a);
}


template <class M>
void
TransformField::callback(Field *ifield, M *)
{
  Field *ofield = ifield->clone();
  ofield->mesh_detach();

  // do our own detach / clone
  M *mesh = (M *)(ofield->mesh().get_rep());
  const int sz = mesh->nodes_size();
  for (int i = 0; i < sz; i++)
  {
    Point p;
    mesh->get_point(p, i);
    mesh->set_point(trans_.project(p), i);
  }

  FieldOPort *ofp = (FieldOPort *)get_oport("Transformed Field");
  FieldHandle fh(ofield);
  ofp->send(fh);
}


void
TransformField::execute()
{
  // Get input field.
  FieldIPort *ifp = (FieldIPort *)get_iport("Input Field");
  FieldHandle ifield_handle;
  Field *ifield;
  if (!(ifp->get(ifield_handle) && (ifield = ifield_handle.get_rep())))
  {
    return;
  }

  MatrixIPort *imp = (MatrixIPort *)get_iport("Transform Matrix");
  MatrixHandle imatrix_handle;
  if (!(imp->get(imatrix_handle)))
  {
    return;
  }
  matrix_to_transform(imatrix_handle, trans_);

  const string geom_name = ifield->get_type_name(0);
  if (geom_name == "TetVol")
  {
    callback(ifield, (TetVolMesh *)0);
  }
  else if (geom_name == "TriSurf")
  {
    callback(ifield, (TriSurfMesh *) 0);
  }
  else if (geom_name == "ContourField")
  {
    callback(ifield, (ContourMesh *) 0);
  }
  else if (geom_name == "PointCloud")
  {
    callback(ifield, (PointCloudMesh *) 0);
  }
  else
  {
    cout << "TransformField:: Unsupported mesh type\n";
  }
}

} // End namespace SCIRun

