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
 *  ManageFieldData: Store/retrieve values from an input matrix to/from 
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
#include <Core/Datatypes/ContourField.h>
#include <Core/Datatypes/PointCloud.h>
#include <Core/Datatypes/TetVol.h>
#include <Core/Datatypes/LatticeVol.h>
#include <Core/Datatypes/TriSurf.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Datatypes/DispatchScalar1.h>
#include <iostream>
#include <stdio.h>

namespace SCIRun {

class ManageFieldData : public Module
{
public:
  ManageFieldData(const string& id);
  virtual ~ManageFieldData();

  template <class F> void callback_scalar(F *ifield);
  template <class IM, class OF> void callback_mesh(IM *m, OF *f);
  virtual void execute();
};


extern "C" Module* make_ManageFieldData(const string& id)
{
  return new ManageFieldData(id);
}

ManageFieldData::ManageFieldData(const string& id)
  : Module("ManageFieldData", id, Filter, "Fields", "SCIRun")
{
}



ManageFieldData::~ManageFieldData()
{
}


template <class F>
void
ManageFieldData::callback_scalar(F *ifield)
{
  MatrixOPort *omatrix_port = (MatrixOPort *)get_oport("Output Matrix");
  if (omatrix_port->nconnections() > 0)
  {
    typename F::mesh_handle_type mesh = ifield->get_typed_mesh();
    ColumnMatrix *omatrix;
    switch (ifield->data_at())
    {
    case Field::NODE:
      {
	int index = 0;
	omatrix = new ColumnMatrix(mesh->nodes_size());
	typename F::mesh_type::node_iterator iter = mesh->node_begin();
	while (iter != mesh->node_end())
	{
	  typename F::value_type val = ifield->value(*iter);
	  omatrix->put(index++, (double)val);
	  ++iter;
	}
      }
      break;

    case Field::EDGE:
      {
	int index = 0;
	omatrix = new ColumnMatrix(mesh->edges_size());
	typename F::mesh_type::edge_iterator iter = mesh->edge_begin();
	while (iter != mesh->edge_end())
	{
	  typename F::value_type val = ifield->value(*iter);
	  omatrix->put(index++, (double)val);
	  ++iter;
	}
      }
      break;

    case Field::FACE:
      {
	int index = 0;
	omatrix = new ColumnMatrix(mesh->faces_size());
	typename F::mesh_type::face_iterator iter = mesh->face_begin();
	while (iter != mesh->face_end())
	{
	  typename F::value_type val = ifield->value(*iter);
	  omatrix->put(index++, (double)val);
	  ++iter;
	}
      }
      break;

    case Field::CELL:
      {
	int index = 0;
	omatrix = new ColumnMatrix(mesh->cells_size());
	typename F::mesh_type::cell_iterator iter = mesh->cell_begin();
	while (iter != mesh->cell_end())
	{
	  typename F::value_type val = ifield->value(*iter);
	  omatrix->put(index++, (double)val);
	  ++iter;
	}
      }
      break;

    case Field::NONE:
      // No data to put in matrix.
      return;
    }
    
    MatrixHandle omatrix_handle(omatrix);
    omatrix_port->send(omatrix_handle);
  }
}


template <class IMesh, class OField>
void
ManageFieldData::callback_mesh(IMesh *imesh, OField *)
{
  FieldOPort *ofield_port = (FieldOPort *)get_oport("Output Field");
  if (ofield_port->nconnections() > 0)
  {
    MatrixIPort *imatrix_port = (MatrixIPort *)get_iport("Input Matrix");
    MatrixHandle imatrix;
    if (!imatrix_port->get(imatrix))
    {
      remark("No input matrix connected.");
      return;
    }

    const unsigned int rows = imatrix->nrows();
    OField *ofield;
    if (rows && rows == (unsigned int)imesh->nodes_size())
    {
      int index = 0;
      ofield = new OField(imesh, Field::NODE);
      typename IMesh::node_iterator iter = imesh->node_begin();
      while (iter != imesh->node_end())
      {
	ofield->set_value(imatrix->get(index++, 0), *iter);
	++iter;
      }
    }
    else if (rows && rows == (unsigned int)imesh->edges_size())
    {
      int index = 0;
      ofield = new OField(imesh, Field::EDGE);
      typename IMesh::edge_iterator iter = imesh->edge_begin();
      while (iter != imesh->edge_end())
      {
	ofield->set_value(imatrix->get(index++, 0), *iter);
	++iter;
      }
    }
    else if (rows && rows == (unsigned int)imesh->faces_size())
    {
      int index = 0;
      ofield = new OField(imesh, Field::FACE);
      typename IMesh::face_iterator iter = imesh->face_begin();
      while (iter != imesh->face_end())
      {
	ofield->set_value(imatrix->get(index++, 0), *iter);
	++iter;
      }
    }
    else if (rows && rows == (unsigned int)imesh->cells_size())
    {
      int index = 0;
      ofield = new OField(imesh, Field::CELL);
      typename IMesh::cell_iterator iter = imesh->cell_begin();
      while (iter != imesh->cell_end())
      {
	ofield->set_value(imatrix->get(index++, 0), *iter);
	++iter;
      }
    }
    else
    {
      error("Matrix datasize does not match field geometry.");
      return;
    }

    FieldHandle fh(ofield);
    ofield_port->send(fh);
  }
}



void
ManageFieldData::execute()
{
  // Get input field.
  FieldIPort *ifp = (FieldIPort *)get_iport("Input Field");
  FieldHandle ifieldhandle;
  Field *ifield; 
  if (!(ifp->get(ifieldhandle) && (ifield = ifieldhandle.get_rep())))
  {
    return;
  }

  // Create a new Vector field with the same geometry handle as field.
  const string geom_name = ifield->get_type_name(0);
  if (geom_name == "TetVol")
  {
    callback_mesh((TetVolMesh *)(ifield->mesh().get_rep()),
		  (TetVol<double> *)0);
  }
  else if (geom_name == "LatticeVol")
  {
    callback_mesh((LatVolMesh *)(ifield->mesh().get_rep()),
		  (LatticeVol<double> *)0);
  }
  else if (geom_name == "TriSurf")
  {
    callback_mesh((TriSurfMesh *)(ifield->mesh().get_rep()),
		  (TriSurf<double> *)0);
  }
  else if (geom_name == "ContourField")
  {
    callback_mesh((ContourMesh *)(ifield->mesh().get_rep()),
		  (ContourField<double> *)0);
  }
  else if (geom_name == "PointCloud")
  {
    callback_mesh((PointCloudMesh *)(ifield->mesh().get_rep()),
		  (PointCloud<double> *)0);
  }
  else
  {
    error("Cannot dispatch on mesh type '" + geom_name + "'.");
    return;
  }

  dispatch_scalar1(ifieldhandle, callback_scalar);
}

} // End namespace SCIRun

