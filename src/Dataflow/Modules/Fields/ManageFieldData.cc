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
#include <Core/Datatypes/ImageField.h>
#include <Core/Datatypes/ScanlineField.h>
#include <Dataflow/Modules/Fields/ManageFieldData.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Datatypes/DenseMatrix.h>
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

  void callback_tetvolvector(TetVol<Vector> *ifield);
  template <class F> void callback_scalar(F *ifield);

  template <class IM, class OF> void callback_mesh_scalar(IM *m, OF *f);
  template <class IM, class OF> void callback_mesh_vector(IM *m, OF *f);
  template <class IM, class OF> void callback_mesh_tensor(IM *m, OF *f);
  virtual void execute();

  template <class Fld, class Loc>
  MatrixHandle callback_scalar1(Fld *ifield, Loc *);

  template <class Fld, class Loc>
  MatrixHandle callback_vector1(Fld *ifield, Loc *);

  template <class Fld, class Loc>
  MatrixHandle callback_tensor1(Fld *ifield, Loc *);
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

void
ManageFieldData::callback_tetvolvector(TetVol<Vector> *ifield)
{
  MatrixOPort *omatrix_port = (MatrixOPort *)get_oport("Output Matrix");
  if (omatrix_port->nconnections() > 0)
  {
    MatrixHandle omh;
    switch (ifield->data_at())
    {
    case Field::NODE:
      {
	omh = callback_vector1(ifield, (TetVolMesh::Node *)0);
      }
      break;

    case Field::EDGE:
      {
	omh = callback_vector1(ifield, (TetVolMesh::Edge *)0);
      }
      break;

    case Field::FACE:
      {
	omh = callback_vector1(ifield, (TetVolMesh::Face *)0);
      }
      break;

    case Field::CELL:
      {
	omh = callback_vector1(ifield, (TetVolMesh::Cell *)0);
      }
      break;

    case Field::NONE:
      // No data to put in matrix.
      return;
    }
    
    omatrix_port->send(omh);
  }
}



template <class Fld, class Loc>
MatrixHandle
ManageFieldData::callback_scalar1(Fld *ifield, Loc *)
{
  typename Fld::mesh_handle_type mesh = ifield->get_typed_mesh();
  ColumnMatrix *omatrix =
    scinew ColumnMatrix(mesh->tsize((typename Loc::size_type *)0));
  int index = 0;
  typename Loc::iterator iter = mesh->tbegin((typename Loc::iterator *)0);
  typename Loc::iterator eiter = mesh->tend((typename Loc::iterator *)0);
  while (iter != eiter)
  {
    typename Fld::value_type val = ifield->value(*iter);
    omatrix->put(index++, (double)val);
    ++iter;
  }

  return MatrixHandle(omatrix);
}


template <class Fld, class Loc>
MatrixHandle
ManageFieldData::callback_vector1(Fld *ifield, Loc *)
{
  typename Fld::mesh_handle_type mesh = ifield->get_typed_mesh();
  DenseMatrix *omatrix =
    scinew DenseMatrix(mesh->tsize((typename Loc::size_type *)0), 3);
  int index = 0;
  typename Loc::iterator iter = mesh->tbegin((typename Loc::iterator *)0);
  typename Loc::iterator eiter = mesh->tend((typename Loc::iterator *)0);
  while (iter != eiter)
  {
    typename Fld::value_type val = ifield->value(*iter);
    (*omatrix)[index][0]=val.x();
    (*omatrix)[index][1]=val.y();
    (*omatrix)[index][2]=val.z();
    index++;
    ++iter;
  }

  return MatrixHandle(omatrix);
}


template <class Fld, class Loc>
MatrixHandle
ManageFieldData::callback_tensor1(Fld *ifield, Loc *)
{
  typename Fld::mesh_handle_type mesh = ifield->get_typed_mesh();
  DenseMatrix *omatrix =
    scinew DenseMatrix(mesh->tsize((typename Loc::size_type *)0), 9);
  int index = 0;
  typename Loc::iterator iter = mesh->tbegin((typename Loc::iterator *)0);
  typename Loc::iterator eiter = mesh->tend((typename Loc::iterator *)0);
  while (iter != eiter)
  {
    typename Fld::value_type val = ifield->value(*iter);
    (*omatrix)[index][0]=val.mat_[0][0];
    (*omatrix)[index][1]=val.mat_[0][1];
    (*omatrix)[index][2]=val.mat_[0][2];;

    (*omatrix)[index][3]=val.mat_[1][0];;
    (*omatrix)[index][4]=val.mat_[1][1];;
    (*omatrix)[index][5]=val.mat_[1][2];;

    (*omatrix)[index][6]=val.mat_[2][0];;
    (*omatrix)[index][7]=val.mat_[2][1];;
    (*omatrix)[index][8]=val.mat_[2][2];;
    index++;
    ++iter;
  }

  return MatrixHandle(omatrix);
}


template <class F>
void
ManageFieldData::callback_scalar(F *ifield)
{
  MatrixOPort *omatrix_port = (MatrixOPort *)get_oport("Output Matrix");
  if (omatrix_port->nconnections() > 0)
  {
    MatrixHandle omh;
    switch (ifield->data_at())
    {
    case Field::NODE:
      {
	omh = callback_scalar1(ifield, (typename F::mesh_type::Node *)0);
      }
      break;

    case Field::EDGE:
      {
	omh = callback_scalar1(ifield, (typename F::mesh_type::Edge *)0);
      }
      break;

    case Field::FACE:
      {
	omh = callback_scalar1(ifield, (typename F::mesh_type::Face *)0);
      }
      break;

    case Field::CELL:
      {
	omh = callback_scalar1(ifield, (typename F::mesh_type::Cell *)0);
      }
      break;

    case Field::NONE:
      // No data to put in matrix.
      return;
    }
    
    omatrix_port->send(omh);
  }
}


template <class IMesh, class OField>
void
ManageFieldData::callback_mesh_scalar(IMesh *imesh, OField *)
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
      ofield = scinew OField(imesh, Field::NODE);
      typename IMesh::Node::iterator iter = imesh->node_begin();
      typename IMesh::Node::iterator eiter = imesh->node_end();
      while (iter != eiter)
      {
	ofield->set_value(imatrix->get(index++, 0), *iter);
	++iter;
      }
    }
    else if (rows && rows == (unsigned int)imesh->edges_size())
    {
      int index = 0;
      ofield = scinew OField(imesh, Field::EDGE);
      typename IMesh::Edge::iterator iter = imesh->edge_begin();
      typename IMesh::Edge::iterator eiter = imesh->edge_end();
      while (iter != eiter)
      {
	ofield->set_value(imatrix->get(index++, 0), *iter);
	++iter;
      }
    }
    else if (rows && rows == (unsigned int)imesh->faces_size())
    {
      int index = 0;
      ofield = scinew OField(imesh, Field::FACE);
      typename IMesh::Face::iterator iter = imesh->face_begin();
      typename IMesh::Face::iterator eiter = imesh->face_end();
      while (iter != eiter)
      {
	ofield->set_value(imatrix->get(index++, 0), *iter);
	++iter;
      }
    }
    else if (rows && rows == (unsigned int)imesh->cells_size())
    {
      int index = 0;
      ofield = scinew OField(imesh, Field::CELL);
      typename IMesh::Cell::iterator iter = imesh->cell_begin();
      typename IMesh::Cell::iterator eiter = imesh->cell_end();
      while (iter != eiter)
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


template <class IMesh, class OField>
void
ManageFieldData::callback_mesh_vector(IMesh *imesh, OField *)
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
      ofield = scinew OField(imesh, Field::NODE);
      typename IMesh::Node::iterator iter = imesh->node_begin();
      typename IMesh::Node::iterator eiter = imesh->node_end();
      while (iter != eiter)
      {
	Vector v(imatrix->get(index, 0),
		 imatrix->get(index, 1),
		 imatrix->get(index, 2));
	index++;
	ofield->set_value(v, *iter);
	++iter;
      }
    }
    else if (rows && rows == (unsigned int)imesh->edges_size())
    {
      int index = 0;
      ofield = scinew OField(imesh, Field::EDGE);
      typename IMesh::Edge::iterator iter = imesh->edge_begin();
      typename IMesh::Edge::iterator eiter = imesh->edge_end();
      while (iter != eiter)
      {
	Vector v(imatrix->get(index, 0),
		 imatrix->get(index, 1),
		 imatrix->get(index, 2));
	index++;
	ofield->set_value(v, *iter);
	++iter;
      }
    }
    else if (rows && rows == (unsigned int)imesh->faces_size())
    {
      int index = 0;
      ofield = scinew OField(imesh, Field::FACE);
      typename IMesh::Face::iterator iter = imesh->face_begin();
      typename IMesh::Face::iterator eiter = imesh->face_end();
      while (iter != eiter)
      {
	Vector v(imatrix->get(index, 0),
		 imatrix->get(index, 1),
		 imatrix->get(index, 2));
	index++;
	ofield->set_value(v, *iter);
	++iter;
      }
    }
    else if (rows && rows == (unsigned int)imesh->cells_size())
    {
      int index = 0;
      ofield = scinew OField(imesh, Field::CELL);
      typename IMesh::Cell::iterator iter = imesh->cell_begin();
      typename IMesh::Cell::iterator eiter = imesh->cell_end();
      while (iter != eiter)
      {
	Vector v(imatrix->get(index, 0),
		 imatrix->get(index, 1),
		 imatrix->get(index, 2));
	index++;
	ofield->set_value(v, *iter);
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


template <class IMesh, class OField>
void
ManageFieldData::callback_mesh_tensor(IMesh *imesh, OField *)
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
      ofield = scinew OField(imesh, Field::NODE);
      typename IMesh::Node::iterator iter = imesh->node_begin();
      typename IMesh::Node::iterator eiter = imesh->node_end();
      while (iter != eiter)
      {
	Tensor v;
	v.mat_[0][0] = imatrix->get(index, 0);
	v.mat_[0][1] = imatrix->get(index, 1);
	v.mat_[0][2] = imatrix->get(index, 2);

	v.mat_[1][0] = imatrix->get(index, 3);
	v.mat_[1][1] = imatrix->get(index, 4);
	v.mat_[1][2] = imatrix->get(index, 5);

	v.mat_[2][0] = imatrix->get(index, 6);
	v.mat_[2][1] = imatrix->get(index, 7);
	v.mat_[2][2] = imatrix->get(index, 8);
	index++;
	ofield->set_value(v, *iter);
	++iter;
      }
    }
    else if (rows && rows == (unsigned int)imesh->edges_size())
    {
      int index = 0;
      ofield = scinew OField(imesh, Field::EDGE);
      typename IMesh::Edge::iterator iter = imesh->edge_begin();
      typename IMesh::Edge::iterator eiter = imesh->edge_end();
      while (iter != eiter)
      {
	Tensor v;
	v.mat_[0][0] = imatrix->get(index, 0);
	v.mat_[0][1] = imatrix->get(index, 1);
	v.mat_[0][2] = imatrix->get(index, 2);

	v.mat_[1][0] = imatrix->get(index, 3);
	v.mat_[1][1] = imatrix->get(index, 4);
	v.mat_[1][2] = imatrix->get(index, 5);

	v.mat_[2][0] = imatrix->get(index, 6);
	v.mat_[2][1] = imatrix->get(index, 7);
	v.mat_[2][2] = imatrix->get(index, 8);
	index++;
	ofield->set_value(v, *iter);
	++iter;
      }
    }
    else if (rows && rows == (unsigned int)imesh->faces_size())
    {
      int index = 0;
      ofield = scinew OField(imesh, Field::FACE);
      typename IMesh::Face::iterator iter = imesh->face_begin();
      typename IMesh::Face::iterator eiter = imesh->face_end();
      while (iter != eiter)
      {
	Tensor v;
	v.mat_[0][0] = imatrix->get(index, 0);
	v.mat_[0][1] = imatrix->get(index, 1);
	v.mat_[0][2] = imatrix->get(index, 2);

	v.mat_[1][0] = imatrix->get(index, 3);
	v.mat_[1][1] = imatrix->get(index, 4);
	v.mat_[1][2] = imatrix->get(index, 5);

	v.mat_[2][0] = imatrix->get(index, 6);
	v.mat_[2][1] = imatrix->get(index, 7);
	v.mat_[2][2] = imatrix->get(index, 8);
	index++;
	ofield->set_value(v, *iter);
	++iter;
      }
    }
    else if (rows && rows == (unsigned int)imesh->cells_size())
    {
      int index = 0;
      ofield = scinew OField(imesh, Field::CELL);
      typename IMesh::Cell::iterator iter = imesh->cell_begin();
      typename IMesh::Cell::iterator eiter = imesh->cell_end();
      while (iter != eiter)
      {
	Tensor v;
	v.mat_[0][0] = imatrix->get(index, 0);
	v.mat_[0][1] = imatrix->get(index, 1);
	v.mat_[0][2] = imatrix->get(index, 2);

	v.mat_[1][0] = imatrix->get(index, 3);
	v.mat_[1][1] = imatrix->get(index, 4);
	v.mat_[1][2] = imatrix->get(index, 5);

	v.mat_[2][0] = imatrix->get(index, 6);
	v.mat_[2][1] = imatrix->get(index, 7);
	v.mat_[2][2] = imatrix->get(index, 8);
	index++;
	ofield->set_value(v, *iter);
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

  if (ifieldhandle->query_scalar_interface())
  {
    // Create a new Vector field with the same geometry handle as field.
    const string geom_name = ifield->get_type_name(0);
    if (geom_name == "TetVol")
    {
      callback_mesh_scalar((TetVolMesh *)(ifield->mesh().get_rep()),
			   (TetVol<double> *)0);
    }
    else if (geom_name == "LatticeVol")
    {
      callback_mesh_scalar((LatVolMesh *)(ifield->mesh().get_rep()),
			   (LatticeVol<double> *)0);
    }
    else if (geom_name == "TriSurf")
    {
      callback_mesh_scalar((TriSurfMesh *)(ifield->mesh().get_rep()),
			   (TriSurf<double> *)0);
    }
    else if (geom_name == "ContourField")
    {
      callback_mesh_scalar((ContourMesh *)(ifield->mesh().get_rep()),
			   (ContourField<double> *)0);
    }
    else if (geom_name == "PointCloud")
    {
      callback_mesh_scalar((PointCloudMesh *)(ifield->mesh().get_rep()),
			   (PointCloud<double> *)0);
    }
    else
    {
      error("Cannot dispatch on mesh type '" + geom_name + "'.");
      return;
    }

    CompileInfo *ci =
      ManageFieldDataAlgoField::get_compile_info(ifieldhandle->get_type_description(),
						 ifieldhandle->data_at_type_description(), 0);
    DynamicAlgoHandle algo_handle;
    if (! DynamicLoader::scirun_loader().get(*ci, algo_handle))
    {
      error("Could not compile algorithm.");
      return;
    }
    ManageFieldDataAlgoField *algo =
      dynamic_cast<ManageFieldDataAlgoField *>(algo_handle.get_rep());
    if (algo == 0)
    {
      error("Could not get algorithm.");
      return;
    }
    algo->execute(ifieldhandle);
  }
  else if (ifieldhandle->query_vector_interface())
  {
    // Create a new Vector field with the same geometry handle as field.
    const string geom_name = ifield->get_type_name(0);
    if (geom_name == "TetVol")
    {
      callback_mesh_vector((TetVolMesh *)(ifield->mesh().get_rep()),
			   (TetVol<Vector> *)0);
    }
    else if (geom_name == "LatticeVol")
    {
      callback_mesh_vector((LatVolMesh *)(ifield->mesh().get_rep()),
			   (LatticeVol<Vector> *)0);
    }
    else if (geom_name == "TriSurf")
    {
      callback_mesh_vector((TriSurfMesh *)(ifield->mesh().get_rep()),
			   (TriSurf<Vector> *)0);
    }
    else if (geom_name == "ContourField")
    {
      callback_mesh_vector((ContourMesh *)(ifield->mesh().get_rep()),
			   (ContourField<Vector> *)0);
    }
    else if (geom_name == "PointCloud")
    {
      callback_mesh_vector((PointCloudMesh *)(ifield->mesh().get_rep()),
			   (PointCloud<Vector> *)0);
    }
    else
    {
      error("Cannot dispatch on mesh type '" + geom_name + "'.");
      return;
    }

    if (ifieldhandle->get_type_name(-1) == "TetVol<Vector>")
    {
      callback_tetvolvector(dynamic_cast<TetVol<Vector> *>(ifieldhandle.get_rep()));
    }
  }
  else if (ifieldhandle->query_tensor_interface())
  {
    // Create a new Tensor field with the same geometry handle as field.
    const string geom_name = ifield->get_type_name(0);
    if (geom_name == "TetVol")
    {
      callback_mesh_tensor((TetVolMesh *)(ifield->mesh().get_rep()),
			   (TetVol<Tensor> *)0);
    }
    else if (geom_name == "LatticeVol")
    {
      callback_mesh_tensor((LatVolMesh *)(ifield->mesh().get_rep()),
			   (LatticeVol<Tensor> *)0);
    }
    else if (geom_name == "TriSurf")
    {
      callback_mesh_tensor((TriSurfMesh *)(ifield->mesh().get_rep()),
			   (TriSurf<Tensor> *)0);
    }
    else if (geom_name == "ContourField")
    {
      callback_mesh_tensor((ContourMesh *)(ifield->mesh().get_rep()),
			   (ContourField<Tensor> *)0);
    }
    else if (geom_name == "PointCloud")
    {
      callback_mesh_tensor((PointCloudMesh *)(ifield->mesh().get_rep()),
			   (PointCloud<Tensor> *)0);
    }
    else
    {
      error("Cannot dispatch on mesh type '" + geom_name + "'.");
      return;
    }

    callback_tensor1(dynamic_cast<TetVol<Tensor> *>(ifieldhandle.get_rep()),
		     (TetVolMesh::Node *)0);
  }
  else
  {
    error("Unable to classify size.");
  }
}


CompileInfo *
ManageFieldDataAlgoField::get_compile_info(const TypeDescription *fsrc,
					   const TypeDescription *lsrc,
					   int svt_flag)
{
  // Use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string base_class_name("ManageFieldDataAlgoField");

  string extension;
  switch (svt_flag)
  {
  case 2:
    extension = "Tensor";
    break;

  case 1:
    extension = "Vector";
    break;

  default:
    extension = "Scalar";
    break;
  }

  CompileInfo *rval = 
    scinew CompileInfo(base_class_name + extension + "." +
		       to_filename(fsrc->get_name()) + "." +
		       to_filename(lsrc->get_name()) + ".",
                       base_class_name, 
                       base_class_name + extension, 
                       fsrc->get_name() + ", " + lsrc->get_name());

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  fsrc->fill_compile_info(rval);
  return rval;
}






} // End namespace SCIRun

