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

#include <Dataflow/Network/Module.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Geometry/Transform.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/MiscMath.h>
#include <iostream>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

namespace SCIRun {


class TransformField : public Module
{
public:
  TransformField(GuiContext* ctx);
  virtual ~TransformField();

  virtual void execute();

  void matrix_to_transform(MatrixHandle mH, Transform& t);
  
protected:
  int ifield_generation_;
  int imatrix_generation_;
};


DECLARE_MAKER(TransformField)

TransformField::TransformField(GuiContext* ctx)
  : Module("TransformField", ctx, Filter, "FieldsGeometry", "SCIRun"),
    ifield_generation_(0),
    imatrix_generation_(0)
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


void
TransformField::execute()
{
  // Get input field.
  FieldIPort *ifp = (FieldIPort *)get_iport("Input Field");
  FieldHandle ifield;
  if (!ifp) {
    error("Unable to initialize iport 'Input Field'.");
    return;
  }
  if (!(ifp->get(ifield) && ifield.get_rep()))
  {
    return;
  }

  MatrixIPort *imp = (MatrixIPort *)get_iport("Transform Matrix");
  MatrixHandle imatrix;
  if (!imp) {
    error("Unable to initialize iport 'Transform Matrix'.");
    return;
  }
  if (!(imp->get(imatrix) && imatrix.get_rep()))
  {
    return;
  }

  if (ifield_generation_ != ifield->generation ||
      imatrix_generation_ != imatrix->generation)
  {
    ifield_generation_ = ifield->generation;
    imatrix_generation_ = imatrix->generation;

    Transform trans;
    matrix_to_transform(imatrix, trans);

    FieldHandle ofield(ifield->clone());
    ofield->mesh_detach();
  
    ofield->mesh()->transform(trans);
    
    FieldOPort *ofp = (FieldOPort *)get_oport("Transformed Field");
    if (!ofp) {
      error("Unable to initialize oport 'Transformed Field'.");
      return;
    }
    ofp->send(ofield);
  }
}


} // End namespace SCIRun

