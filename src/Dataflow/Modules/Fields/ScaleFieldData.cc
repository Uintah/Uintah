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
 *  ScaleFieldData: Store/retrieve values from an input matrix to/from 
 *            the data of a field
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   February 2001
 *
 *  Copyright (C) 2001 SCI Institute
 */

#include <Core/Persistent/Pstreams.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Datatypes/PointCloud.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/GuiInterface/GuiVar.h>

#include <iostream>
#include <stdio.h>

namespace SCIRun {

class ScaleFieldData : public Module
{
public:
  ScaleFieldData(const string& id);
  virtual ~ScaleFieldData();
  virtual void execute();
};


extern "C" Module* make_ScaleFieldData(const string& id)
{
  return new ScaleFieldData(id);
}

ScaleFieldData::ScaleFieldData(const string& id)
  : Module("ScaleFieldData", id, Filter, "Fields", "SCIRun")
{
}



ScaleFieldData::~ScaleFieldData()
{
}


void
ScaleFieldData::execute()
{
  // Get input field.
  FieldIPort *ifp = (FieldIPort *)get_iport("Input Field");
  FieldHandle ifieldhandle;
  Field *ifield; 
  if (!(ifp->get(ifieldhandle) && (ifield = ifieldhandle.get_rep())))
  {
    return;
  }

  MatrixIPort *imatrix_port = (MatrixIPort *)get_iport("Input Matrix");
  MatrixHandle imatrix;
  if (!imatrix_port->get(imatrix))
  {
    return;
  }
  const unsigned int rows = imatrix->nrows();

  // Create a new Vector field with the same geometry handle as field.
  const string geom_name = ifield->get_type_name(0);
  const string data_name = ifield->get_type_name(1);

  if (data_name == "Vector" && geom_name == "PointCloud") {
    PointCloud<Vector> *pc = dynamic_cast<PointCloud<Vector> *>(ifieldhandle.get_rep());
    if (rows != pc->fdata().size()) {
      error("Different size data vectors.");
      return;
    }
    PointCloud<Vector> *pcs = pc->clone();
    for (unsigned int i=0; i<pc->fdata().size(); i++) {
      pcs->fdata()[i] = pc->fdata()[i] * imatrix->get(i, 0);
    }
    FieldOPort *ofield_port = (FieldOPort *)get_oport("Output Field");
    FieldHandle ofieldhandle(pcs);
    ofield_port->send(ofieldhandle);
  } else {
    error("I only know how to scale vectors.");
    return;
  }
}

} // End namespace SCIRun
