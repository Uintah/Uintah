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
#include <Core/Datatypes/DispatchNonlattice1.h>
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
  template <class F> void dispatch(F *ifield);
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

template <class F>
void
TransformField::dispatch(F *ifield)
{
  // do our own detach / clone
  LockingHandle<typename F::mesh_type> omesh = ifield->get_typed_mesh();
  omesh.detach();
  F *ofield = scinew F(omesh, ifield->data_at());
  ofield->fdata() = ifield->fdata();
  int sz = omesh->nodes_size();
  for (int i = 0; i < sz; i++) {
    Point p;
    omesh->get_point(p, i);
    omesh->set_point(trans_.project(p), i);
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
  if (!(ifp->get(ifield_handle) && ifield_handle.get_rep()))
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

  if (ifield_handle->get_type_name(0) != "LatticeVol") {
    dispatch_nonlattice1(ifield_handle, dispatch);
  }
}

} // End namespace SCIRun

