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

  template <class M> void dispatch(M *meshtype, Field *f, Transform &t);
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
TransformField::dispatch(M *, Field *ifield, Transform &trans)
{
  Field *ofield = ifield->clone();

  ofield->mesh().detach();
  M *omesh = (M *)(ofield->mesh().get_rep());
  
  typename M::node_iterator ni = omesh->node_begin();
  while (ni != omesh->node_end())
  {
    Point p;
    omesh->get_point(p, *ni);
    omesh->set_point(trans.project(p), *ni);
    ++ni;
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
  Transform transform;
  matrix_to_transform(imatrix_handle, transform);

  // Create a new Vector field with the same geometry handle as field.
  const string geom_name = ifield->get_type_name(0);
  const string data_name = ifield->get_type_name(1);
  if (geom_name == "TetVol")
  {
    dispatch((TetVolMesh *)0, ifield, transform);
  }
  else if (geom_name == "LatticeVol")
  {
    // Error cannot transform these yet.
  }
  else if (geom_name == "TriSurf")
  {
    dispatch((TriSurfMesh *)0, ifield, transform);
  }
  else
  {
    // Don't know what to do with this field type.
    // Signal some sort of error.
    return;
  }
}

} // End namespace SCIRun

