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

#include <iostream>
#include <stdio.h>

namespace SCIRun {

class ManageFieldData : public Module
{
public:
  ManageFieldData(const string& id);
  virtual ~ManageFieldData();

  template <class F> void dispatch_scalar(F *ifield);
  template <class F> void dispatch_tetvol(F *ifield);
  template <class F> void dispatch_latticevol(F *ifield);
  template <class F> void dispatch_trisurf(F *ifield);
  template <class F> void dispatch_contourfield(F *ifield);
  template <class F> void dispatch_pointcloud(F *ifield);
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
ManageFieldData::dispatch_scalar(F *ifield)
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


template <class F>
void
ManageFieldData::dispatch_tetvol(F *ifield)
{
  FieldOPort *ofield_port = (FieldOPort *)get_oport("Output Field");
  if (ofield_port->nconnections() > 0)
  {
    MatrixIPort *imatrix_port = (MatrixIPort *)get_iport("Input Matrix");
    MatrixHandle imatrix;
    if (!imatrix_port->get(imatrix))
    {
      return;
    }

    const int rows = imatrix->nrows();
    TetVolMeshHandle mesh = ifield->get_typed_mesh();
    TetVol<double> *ofield;
    if (rows == mesh->nodes_size())
    {
      int index = 0;
      ofield = new TetVol<double>(mesh, Field::NODE);
      typename F::mesh_type::node_iterator iter = mesh->node_begin();
      while (iter != mesh->node_end())
      {
	ofield->set_value(imatrix->get(index++, 0), *iter);
	++iter;
      }
    }
    else if (rows == mesh->edges_size())
    {
      int index = 0;
      ofield = new TetVol<double>(mesh, Field::EDGE);
      typename F::mesh_type::edge_iterator iter = mesh->edge_begin();
      while (iter != mesh->edge_end())
      {
	ofield->set_value(imatrix->get(index++, 0), *iter);
	++iter;
      }
    }
    else if (rows == mesh->faces_size())
    {
      int index = 0;
      ofield = new TetVol<double>(mesh, Field::FACE);
      typename F::mesh_type::face_iterator iter = mesh->face_begin();
      while (iter != mesh->face_end())
      {
	ofield->set_value(imatrix->get(index++, 0), *iter);
	++iter;
      }
    }
    else if (rows == mesh->cells_size())
    {
      int index = 0;
      ofield = new TetVol<double>(mesh, Field::CELL);
      typename F::mesh_type::cell_iterator iter = mesh->cell_begin();
      while (iter != mesh->cell_end())
      {
	ofield->set_value(imatrix->get(index++, 0), *iter);
	++iter;
      }
    }
    else
    {
      // ERROR, matrix datasize does not match field geometry.
      return;
    }

    FieldHandle fh(ofield);
    ofield_port->send(fh);
  }
}


template <class F>
void
ManageFieldData::dispatch_latticevol(F *ifield)
{
  FieldOPort *ofield_port = (FieldOPort *)get_oport("Output Field");
  if (ofield_port->nconnections() > 0)
  {
    MatrixIPort *imatrix_port = (MatrixIPort *)get_iport("Input Matrix");
    MatrixHandle imatrix;
    if (!imatrix_port->get(imatrix))
    {
      return;
    }

    const unsigned int rows = imatrix->nrows();
    LatVolMeshHandle mesh = ifield->get_typed_mesh();
    LatticeVol<double> *ofield;
    if (rows == mesh->nodes_size())
    {
      int index = 0;
      ofield = new LatticeVol<double>(mesh, Field::NODE);
      typename F::mesh_type::node_iterator iter = mesh->node_begin();
      while (iter != mesh->node_end())
      {
	ofield->set_value(imatrix->get(index++, 0), *iter);
	++iter;
      }
    }
    else if (rows == mesh->edges_size())
    {
      int index = 0;
      ofield = new LatticeVol<double>(mesh, Field::EDGE);
      typename F::mesh_type::edge_iterator iter = mesh->edge_begin();
      while (iter != mesh->edge_end())
      {
	ofield->set_value(imatrix->get(index++, 0), *iter);
	++iter;
      }
    }
    else if (rows == mesh->faces_size())
    {
      int index = 0;
      ofield = new LatticeVol<double>(mesh, Field::FACE);
      typename F::mesh_type::face_iterator iter = mesh->face_begin();
      while (iter != mesh->face_end())
      {
	ofield->set_value(imatrix->get(index++, 0), *iter);
	++iter;
      }
    }
    else if (rows == mesh->cells_size())
    {
      int index = 0;
      ofield = new LatticeVol<double>(mesh, Field::CELL);
      typename F::mesh_type::cell_iterator iter = mesh->cell_begin();
      while (iter != mesh->cell_end())
      {
	ofield->set_value(imatrix->get(index++, 0), *iter);
	++iter;
      }
    }
    else
    {
      // ERROR, matrix datasize does not match field geometry.
      return;
    }

    FieldHandle fh(ofield);
    ofield_port->send(fh);
  }
}


template <class F>
void
ManageFieldData::dispatch_trisurf(F *ifield)
{
  FieldOPort *ofield_port = (FieldOPort *)get_oport("Output Field");
  if (ofield_port->nconnections() > 0)
  {
    MatrixIPort *imatrix_port = (MatrixIPort *)get_iport("Input Matrix");
    MatrixHandle imatrix;
    if (!imatrix_port->get(imatrix))
    {
      return;
    }

    const int rows = imatrix->nrows();
    TriSurfMeshHandle mesh = ifield->get_typed_mesh();
    TriSurf<double> *ofield;
    if (rows == mesh->nodes_size())
    {
      int index = 0;
      ofield = new TriSurf<double>(mesh, Field::NODE);
      typename F::mesh_type::node_iterator iter = mesh->node_begin();
      while (iter != mesh->node_end())
      {
	ofield->set_value(imatrix->get(index++, 0), *iter);
	++iter;
      }
    }
    else if (rows == mesh->edges_size())
    {
      int index = 0;
      ofield = new TriSurf<double>(mesh, Field::EDGE);
      typename F::mesh_type::edge_iterator iter = mesh->edge_begin();
      while (iter != mesh->edge_end())
      {
	ofield->set_value(imatrix->get(index++, 0), *iter);
	++iter;
      }
    }
    else if (rows == mesh->faces_size())
    {
      int index = 0;
      ofield = new TriSurf<double>(mesh, Field::FACE);
      typename F::mesh_type::face_iterator iter = mesh->face_begin();
      while (iter != mesh->face_end())
      {
	ofield->set_value(imatrix->get(index++, 0), *iter);
	++iter;
      }
    }
    else
    {
      // ERROR, matrix datasize does not match field geometry.
      return;
    }

    FieldHandle fh(ofield);
    ofield_port->send(fh);
  }
}



template <class F>
void
ManageFieldData::dispatch_pointcloud(F *ifield)
{
  FieldOPort *ofield_port = (FieldOPort *)get_oport("Output Field");
  if (ofield_port->nconnections() > 0)
  {
    MatrixIPort *imatrix_port = (MatrixIPort *)get_iport("Input Matrix");
    MatrixHandle imatrix;
    if (!imatrix_port->get(imatrix))
    {
      return;
    }

    const unsigned int rows = imatrix->nrows();
    PointCloudMeshHandle mesh = ifield->get_typed_mesh();
    PointCloud<double> *ofield;
    if (rows == mesh->nodes_size())
    {
      int index = 0;
      ofield = new PointCloud<double>(mesh, Field::NODE);
      typename F::mesh_type::node_iterator iter = mesh->node_begin();
      while (iter != mesh->node_end())
      {
	ofield->set_value(imatrix->get(index++, 0), *iter);
	++iter;
      }
    }
    else if (rows == mesh->edges_size())
    {
      int index = 0;
      ofield = new PointCloud<double>(mesh, Field::EDGE);
      typename F::mesh_type::edge_iterator iter = mesh->edge_begin();
      while (iter != mesh->edge_end())
      {
	ofield->set_value(imatrix->get(index++, 0), *iter);
	++iter;
      }
    }
    else if (rows == mesh->faces_size())
    {
      int index = 0;
      ofield = new PointCloud<double>(mesh, Field::FACE);
      typename F::mesh_type::face_iterator iter = mesh->face_begin();
      while (iter != mesh->face_end())
      {
	ofield->set_value(imatrix->get(index++, 0), *iter);
	++iter;
      }
    }
    else
    {
      // ERROR, matrix datasize does not match field geometry.
      return;
    }

    FieldHandle fh(ofield);
    ofield_port->send(fh);
  }
}


template <class F>
void
ManageFieldData::dispatch_contourfield(F *ifield)
{
  FieldOPort *ofield_port = (FieldOPort *)get_oport("Output Field");
  if (ofield_port->nconnections() > 0)
  {
    MatrixIPort *imatrix_port = (MatrixIPort *)get_iport("Input Matrix");
    MatrixHandle imatrix;
    if (!imatrix_port->get(imatrix))
    {
      return;
    }

    const unsigned int rows = imatrix->nrows();
    ContourMeshHandle mesh = ifield->get_typed_mesh();
    ContourField<double> *ofield;
    if (rows == mesh->nodes_size())
    {
      int index = 0;
      ofield = new ContourField<double>(mesh, Field::NODE);
      typename F::mesh_type::node_iterator iter = mesh->node_begin();
      while (iter != mesh->node_end())
      {
	ofield->set_value(imatrix->get(index++, 0), *iter);
	++iter;
      }
    }
    else if (rows == mesh->edges_size())
    {
      int index = 0;
      ofield = new ContourField<double>(mesh, Field::EDGE);
      typename F::mesh_type::edge_iterator iter = mesh->edge_begin();
      while (iter != mesh->edge_end())
      {
	ofield->set_value(imatrix->get(index++, 0), *iter);
	++iter;
      }
    }
    else if (rows == mesh->faces_size())
    {
      int index = 0;
      ofield = new ContourField<double>(mesh, Field::FACE);
      typename F::mesh_type::face_iterator iter = mesh->face_begin();
      while (iter != mesh->face_end())
      {
	ofield->set_value(imatrix->get(index++, 0), *iter);
	++iter;
      }
    }
    else
    {
      // ERROR, matrix datasize does not match field geometry.
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
  const string data_name = ifield->get_type_name(1);
  if (geom_name == "TetVol")
  {
    if (data_name == "double")
    {
      dispatch_scalar((TetVol<double> *)ifield);
      dispatch_tetvol((TetVol<double> *)ifield);
    }
    else if (data_name == "int")
    {
      dispatch_scalar((TetVol<int> *)ifield);
      dispatch_tetvol((TetVol<int> *)ifield);
    }
    else if (data_name == "short")
    {
      dispatch_scalar((TetVol<short> *)ifield);
      dispatch_tetvol((TetVol<short> *)ifield);
    }
    else if (data_name == "char")
    {
      dispatch_scalar((TetVol<char> *)ifield);
      dispatch_tetvol((TetVol<char> *)ifield);
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
      dispatch_scalar((LatticeVol<double> *)ifield);
      dispatch_latticevol((LatticeVol<double> *)ifield);
    }
    else if (data_name == "int")
    {
      dispatch_scalar((LatticeVol<int> *)ifield);
      dispatch_latticevol((LatticeVol<int> *)ifield);
    }
    else if (data_name == "short")
    {
      dispatch_scalar((LatticeVol<short> *)ifield);
      dispatch_latticevol((LatticeVol<short> *)ifield);
    }
    else if (data_name == "char")
    {
      dispatch_scalar((LatticeVol<char> *)ifield);
      dispatch_latticevol((LatticeVol<char> *)ifield);
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
      dispatch_scalar((TriSurf<double> *)ifield);
      dispatch_trisurf((TriSurf<double> *)ifield);
    }
    else if (data_name == "int")
    {
      dispatch_scalar((TriSurf<int> *)ifield);
      dispatch_trisurf((TriSurf<int> *)ifield);
    }
    else if (data_name == "short")
    {
      dispatch_scalar((TriSurf<short> *)ifield);
      dispatch_trisurf((TriSurf<short> *)ifield);
    }
    else if (data_name == "char")
    {
      dispatch_scalar((TriSurf<char> *)ifield);
      dispatch_trisurf((TriSurf<char> *)ifield);
    }
    else
    {
      // Don't know what to do with this field type.
      // Signal some sort of error.
    }
  }
  else if (geom_name == "PointCloud")
  {
    if (data_name == "double")
    {
      dispatch_scalar((PointCloud<double> *)ifield);
      dispatch_pointcloud((PointCloud<double> *)ifield);
    }
    else if (data_name == "int")
    {
      dispatch_scalar((PointCloud<int> *)ifield);
      dispatch_pointcloud((PointCloud<int> *)ifield);
    }
    else if (data_name == "short")
    {
      dispatch_scalar((PointCloud<short> *)ifield);
      dispatch_pointcloud((PointCloud<short> *)ifield);
    }
    else if (data_name == "char")
    {
      dispatch_scalar((PointCloud<char> *)ifield);
      dispatch_pointcloud((PointCloud<char> *)ifield);
    }
    else
    {
      // Don't know what to do with this field type.
      // Signal some sort of error.
    }
  }
  else if (geom_name == "ContourField")
  {
    if (data_name == "double")
    {
      dispatch_scalar((ContourField<double> *)ifield);
      dispatch_contourfield((ContourField<double> *)ifield);
    }
    else if (data_name == "Vector")
    {
//      dispatch_scalar((ContourField<int> *)ifield);
      dispatch_contourfield((ContourField<Vector> *)ifield);
    }
    else if (data_name == "int")
    {
      dispatch_scalar((ContourField<int> *)ifield);
      dispatch_contourfield((ContourField<int> *)ifield);
    }
    else if (data_name == "short")
    {
      dispatch_scalar((ContourField<short> *)ifield);
      dispatch_contourfield((ContourField<short> *)ifield);
    }
    else if (data_name == "char")
    {
      dispatch_scalar((ContourField<char> *)ifield);
      dispatch_contourfield((ContourField<char> *)ifield);
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

