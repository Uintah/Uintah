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

#include <Core/Containers/String.h>
#include <Core/Persistent/Pstreams.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Datatypes/TetVol.h>
#include <Core/Datatypes/LatticeVol.h>
#include <Core/Datatypes/TriSurf.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/GuiInterface/GuiVar.h>

#include <iostream>
using std::cerr;
#include <stdio.h>

namespace SCIRun {

class ManageFieldData : public Module
{
public:
  ManageFieldData(const clString& id);
  virtual ~ManageFieldData();

  template <class F> void dispatch_scalar(F *ifield);
  template <class F> void dispatch_tetvol(F *ifield);
  template <class F> void dispatch_latticevol(F *ifield);
  template <class F> void dispatch_trisurf(F *ifield);
  virtual void execute();
};


extern "C" Module* make_ManageFieldData(const clString& id)
{
  return new ManageFieldData(id);
}

ManageFieldData::ManageFieldData(const clString& id)
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
    int rows;
    typename F::mesh_handle_type mesh = ifield->get_typed_mesh();
    switch (ifield->data_at())
    {
    case Field::NODE:
      rows = mesh->nodes_size();
      break;
    case Field::EDGE:
      rows = mesh->edges_size();
      break;
    case Field::FACE:
      rows = mesh->faces_size();
      break;
    case Field::CELL:
      rows = mesh->cells_size();
      break;
    case Field::NONE:
      // No data to put in matrix.
      return;
    }

    int index = 0;
    ColumnMatrix *omatrix = new ColumnMatrix(rows);
    typename F::mesh_type::cell_iterator ci = mesh->cell_begin();
    while (ci != mesh->cell_end())
    {
      typename F::value_type val = ifield->value(*ci);
      omatrix->put(index++, (double)val);
      ++ci;
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
    TetVolMeshHandle tvm = ifield->get_typed_mesh();
    Field::data_location loc = Field::NONE;
    if (rows == tvm->nodes_size())
    {
      loc = Field::NODE;
    }
    else if (rows == tvm->edges_size())
    {
      loc = Field::EDGE;
    }
    else if (rows == tvm->faces_size())
    {
      loc = Field::FACE;
    }
    else if (rows == tvm->cells_size())
    {
      loc = Field::CELL;
    }

    if (rows == 0 || loc == Field::NONE)
    {
      // ERROR, matrix datasize does not match field geometry.
      return;
    }

    int index = 0;
    TetVol<double> *ofield = new TetVol<double>(tvm, loc);
    typename F::mesh_type::cell_iterator ci = tvm->cell_begin();
    while (ci != tvm->cell_end())
    {
      ofield->set_value(imatrix->get(index++, 0), *ci);
      ++ci;
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

    unsigned int rows = imatrix->nrows();
    LatVolMeshHandle lvm = ifield->get_typed_mesh();
    Field::data_location loc = Field::NONE;
    if (rows == lvm->nodes_size())
    {
      loc = Field::NODE;
    }
    else if (rows == lvm->edges_size())
    {
      loc = Field::EDGE;
    }
    else if (rows == lvm->faces_size())
    {
      loc = Field::FACE;
    }
    else if (rows == lvm->cells_size())
    {
      loc = Field::CELL;
    }

    if (rows == 0 || loc == Field::NONE)
    {
      // ERROR, matrix datasize does not match field geometry.
      return;
    }

    int index = 0;
    LatticeVol<double> *ofield = new LatticeVol<double>(lvm, loc);
    typename F::mesh_type::cell_iterator ci = lvm->cell_begin();
    while (ci != lvm->cell_end())
    {
      ofield->set_value(imatrix->get(index++, 0), *ci);
      ++ci;
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

    int rows = imatrix->nrows();
    Field::data_location loc = Field::NONE;
    TriSurfMeshHandle tsm = ifield->get_typed_mesh();
    if (rows == tsm->nodes_size())
    {
      loc = Field::NODE;
    }
    else if (rows == tsm->edges_size())
    {
      loc = Field::EDGE;
    }
    else if (rows == tsm->faces_size())
    {
      loc = Field::FACE;
    }
    else if (rows == tsm->cells_size())
    {
      loc = Field::CELL;
    }

    if (rows == 0 || loc == Field::NONE)
    {
      // ERROR, matrix datasize does not match field geometry.
      return;
    }

    int index = 0;
    TriSurf<double> *ofield = new TriSurf<double>(tsm, loc);
    typename F::mesh_type::cell_iterator ci = tsm->cell_begin();
    while (ci != tsm->cell_end())
    {
      ofield->set_value(imatrix->get(index++, 0), *ci);
      ++ci;
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
  else
  {
    // Don't know what to do with this field type.
    // Signal some sort of error.
    return;
  }
}

} // End namespace SCIRun
