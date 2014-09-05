/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   
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
 *  TransformMeshWithTransform.cc:  Rotate and flip field to get it into "standard" view
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   March 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Dataflow/Network/Ports/MatrixPort.h>
#include <Dataflow/Network/Ports/FieldPort.h>
#include <Core/Geometry/Transform.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/MiscMath.h>
#include <iostream>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

namespace SCIRun {


class TransformMeshWithTransform : public Module
{
public:
  TransformMeshWithTransform(GuiContext* ctx);
  virtual ~TransformMeshWithTransform();

  virtual void execute();

  void matrix_to_transform(MatrixHandle mH, Transform& t);
  
protected:
  int ifield_generation_;
  int imatrix_generation_;
};


DECLARE_MAKER(TransformMeshWithTransform)

TransformMeshWithTransform::TransformMeshWithTransform(GuiContext* ctx)
  : Module("TransformMeshWithTransform", ctx, Filter, "ChangeMesh", "SCIRun"),
    ifield_generation_(0),
    imatrix_generation_(0)
{
}


TransformMeshWithTransform::~TransformMeshWithTransform()
{
}


void
TransformMeshWithTransform::matrix_to_transform(MatrixHandle mH, Transform& t)
{
  double a[16];
  double *p=&(a[0]);
  for (int i=0; i<4; i++)
    for (int j=0; j<4; j++)
      *p++= mH->get(i, j);
  t.set(a);
}


void
TransformMeshWithTransform::execute()
{
  // Get input field.
  FieldHandle ifield;
  if (!get_input_handle("Input Field", ifield)) return;

  MatrixHandle imatrix;
  if (!get_input_handle("Transform Matrix", imatrix)) return;

  if (ifield_generation_ != ifield->generation ||
      imatrix_generation_ != imatrix->generation ||
      !oport_cached("Transformed Field"))
  {
    ifield_generation_ = ifield->generation;
    imatrix_generation_ = imatrix->generation;

    Transform trans;
    matrix_to_transform(imatrix, trans);

    FieldHandle ofield(ifield->clone());
    ofield->mesh_detach();
  
    ofield->mesh()->transform(trans);
    
    send_output_handle("Transformed Field", ofield);
  }
}


} // End namespace SCIRun

