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

  template <class F> void dispatch_tetvol(F *f, Transform &t);
  template <class F> void dispatch_latticevol(F *f, Transform &t);
  template <class F> void dispatch_trisurf(F *f, Transform &t);
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
TransformField::dispatch_tetvol(F *ifield, Transform &trans)
{
  F *ofield = (F *)ifield->clone();

  typename F::mesh_type *omesh =
    (typename F::mesh_type *)ofield->get_typed_mesh()->clone();
  
  typename F::mesh_type::node_iterator ni = omesh->node_begin();
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


template <class F>
void
TransformField::dispatch_latticevol(F *ifield, Transform &trans)
{
  
}


template <class F>
void
TransformField::dispatch_trisurf(F *ifield, Transform &trans)
{
  F *ofield = (F *)ifield->clone();

  typename F::mesh_type *omesh =
    (typename F::mesh_type *)ofield->get_typed_mesh()->clone();

  typename F::mesh_type::node_iterator ni = omesh->node_begin();
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
    if (data_name == "double")
    {
      dispatch_tetvol((TetVol<double> *)ifield, transform);
    }
    else if (data_name == "int")
    {
      dispatch_tetvol((TetVol<int> *)ifield, transform);
    }
    else if (data_name == "short")
    {
      dispatch_tetvol((TetVol<short> *)ifield, transform);
    }
    else if (data_name == "char")
    {
      dispatch_tetvol((TetVol<char> *)ifield, transform);
    }
    else
    {
      // Don't know what to do with this field type.
      // Signal some sort of error.
    }
  }
  else if (geom_name == "LatticeVol")
  {
    if (data_name == "double")
    {
      dispatch_latticevol((LatticeVol<double> *)ifield, transform);
    }
    else if (data_name == "int")
    {
      dispatch_latticevol((LatticeVol<int> *)ifield, transform);
    }
    else if (data_name == "short")
    {
      dispatch_latticevol((LatticeVol<short> *)ifield, transform);
    }
    else if (data_name == "char")
    {
      dispatch_latticevol((LatticeVol<char> *)ifield, transform);
    }
    else
    {
      // Don't know what to do with this field type.
      // Signal some sort of error.
    }
  }
  else if (geom_name == "TriSurf")
  {
    if (data_name == "double")
    {
      dispatch_trisurf((TriSurf<double> *)ifield, transform);
    }
    else if (data_name == "int")
    {
      dispatch_trisurf((TriSurf<int> *)ifield, transform);
    }
    else if (data_name == "short")
    {
      dispatch_trisurf((TriSurf<short> *)ifield, transform);
    }
    else if (data_name == "char")
    {
      dispatch_trisurf((TriSurf<char> *)ifield, transform);
    }
    else
    {
      // Don't know what to do with this field type.
      // Signal some sort of error.
    }
  }
  else
  {
    // Don't know what to do with this field type.
    // Signal some sort of error.
    return;
  }
}

} // End namespace SCIRun

