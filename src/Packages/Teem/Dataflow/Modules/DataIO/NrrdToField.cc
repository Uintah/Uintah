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
 *  NrrdToField.cc:  Convert a Nrrd to a Field
 *
 *  Written by:
 *   David Weinstein
 *   School of Computing
 *   University of Utah
 *   February 2001
 *
 *  Copyright (C) 2001 SCI Institute
 */

#include <Dataflow/Network/Module.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Teem/Dataflow/Ports/NrrdPort.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Datatypes/MaskedLatVolField.h>
#include <Core/Datatypes/ScanlineField.h>
#include <Core/Malloc/Allocator.h>
#include <Teem/Dataflow/Modules/DataIO/ConvertToField.h>
#include <Core/Util/TypeDescription.h>
#include <iostream>

namespace SCITeem {

using namespace SCIRun;

class NrrdToField : public Module {
  NrrdIPort* inrrd;
  FieldOPort* ofield;
public:
  NrrdToField(GuiContext *ctx);
  virtual ~NrrdToField();
  virtual void execute();
private:
  FieldHandle create_scanline_field(NrrdDataHandle &nrd);
  FieldHandle create_image_field(NrrdDataHandle &nrd);
  FieldHandle create_latvol_field(NrrdDataHandle &nrd);
  GuiInt build_eigens_;
};

} // end namespace SCITeem

using namespace SCITeem;
DECLARE_MAKER(NrrdToField)

NrrdToField::NrrdToField(GuiContext *ctx):
  Module("NrrdToField", ctx, Filter, "DataIO", "Teem"),
  build_eigens_(ctx->subVar("build-eigens"))
{
}

NrrdToField::~NrrdToField()
{
}

FieldHandle 
NrrdToField::create_scanline_field(NrrdDataHandle &nrd) 
{
  Nrrd *n = nrd->nrrd;

  double spc;
  if ( AIR_EXISTS(n->axis[1].spacing)) { spc = n->axis[1].spacing; }
  else { spc = 1.; }
  int data_center = n->axis[1].center;
  for (int a = 1; a < 2; a++) {
    if (!(AIR_EXISTS(n->axis[a].min) && AIR_EXISTS(n->axis[a].max)))
      nrrdAxisMinMaxSet(n, a, nrrdCenterNode);
  }

  // if nothing was specified, just call it node-centered (arbitrary)
  if (data_center == nrrdCenterUnknown) data_center = nrrdCenterNode;

  Point min(0., 0., 0.);
  Point max;
  
  if (data_center == nrrdCenterCell) {
    max = Point(n->axis[1].size * spc, 
		0.0, 0.0);
  } else {
    max = Point((n->axis[1].size - 1) * spc, 
		0.0, 0.0);
  }
  int off = 0;
  if (data_center == nrrdCenterCell) { off = 1; }
  ScanlineMesh *m = new ScanlineMesh(n->axis[1].size + off, min, max);
  ScanlineMeshHandle mh(m);
  FieldHandle fh;

  int mn_idx, mx_idx;
  nrd->get_tuple_index_info(0, 0, mn_idx, mx_idx);
  
  switch (mx_idx) {
  case 0:
    switch (n->type) {
    case nrrdTypeChar :  
      if (data_center == nrrdCenterCell) {
	fh = new ScanlineField<char>(mh, Field::EDGE);
	ScanlineMesh::Edge::iterator iter, end;
	mh->begin(iter);
	mh->end(end);
	fill_data((ScanlineField<char>*)fh.get_rep(), n, iter, end);
      } else {
	fh = new ScanlineField<char>(mh, Field::NODE);
	ScanlineMesh::Node::iterator iter, end;
	mh->begin(iter);
	mh->end(end);
	fill_data((ScanlineField<char>*)fh.get_rep(), n, iter, end);
      }

      break;
    case nrrdTypeUChar : 
      if (data_center == nrrdCenterCell) {
	fh = new ScanlineField<unsigned char>(mh, Field::EDGE);
	ScanlineMesh::Edge::iterator iter, end;
	mh->begin(iter);
	mh->end(end);
	fill_data((ScanlineField<unsigned char>*)fh.get_rep(), n, iter, end);
      } else {
	fh = new ScanlineField<unsigned char>(mh, Field::NODE);
	ScanlineMesh::Node::iterator iter, end;
	mh->begin(iter);
	mh->end(end);
	fill_data((ScanlineField<unsigned char>*)fh.get_rep(), n, iter, end);
      }

      break;
    case nrrdTypeShort : 
      if (data_center == nrrdCenterCell) {
	fh = new ScanlineField<short>(mh, Field::EDGE);
	ScanlineMesh::Edge::iterator iter, end;
	mh->begin(iter);
	mh->end(end);
	fill_data((ScanlineField<short>*)fh.get_rep(), n, iter, end);
      } else {
	fh = new ScanlineField<short>(mh, Field::NODE);
	ScanlineMesh::Node::iterator iter, end;
	mh->begin(iter);
	mh->end(end);
	fill_data((ScanlineField<short>*)fh.get_rep(), n, iter, end);
      }

      break;
    case nrrdTypeUShort :
      if (data_center == nrrdCenterCell) {
	fh = new ScanlineField<unsigned short>(mh, Field::EDGE);
	ScanlineMesh::Edge::iterator iter, end;
	mh->begin(iter);
	mh->end(end);
	fill_data((ScanlineField<unsigned short>*)fh.get_rep(), n, iter, end);
      } else {
	fh = new ScanlineField<unsigned short>(mh, Field::NODE);
	ScanlineMesh::Node::iterator iter, end;
	mh->begin(iter);
	mh->end(end);
	fill_data((ScanlineField<unsigned short>*)fh.get_rep(), n, iter, end);
      }

      break;
    case nrrdTypeInt : 
      if (data_center == nrrdCenterCell) {
	fh = new ScanlineField<int>(mh, Field::EDGE);
	ScanlineMesh::Edge::iterator iter, end;
	mh->begin(iter);
	mh->end(end);
	fill_data((ScanlineField<int>*)fh.get_rep(), n, iter, end);
      } else {
	fh = new ScanlineField<int>(mh, Field::NODE);
	ScanlineMesh::Node::iterator iter, end;
	mh->begin(iter);
	mh->end(end);
	fill_data((ScanlineField<int>*)fh.get_rep(), n, iter, end);
      }

      break;
    case nrrdTypeUInt :  
      if (data_center == nrrdCenterCell) {
	fh = new ScanlineField<unsigned int>(mh, Field::EDGE);
	ScanlineMesh::Edge::iterator iter, end;
	mh->begin(iter);
	mh->end(end);
	fill_data((ScanlineField<unsigned int>*)fh.get_rep(), n, iter, end);
      } else {
	fh = new ScanlineField<unsigned int>(mh, Field::NODE);
	ScanlineMesh::Node::iterator iter, end;
	mh->begin(iter);
	mh->end(end);
	fill_data((ScanlineField<unsigned int>*)fh.get_rep(), n, iter, end);
      }

      break;
    case nrrdTypeLLong : 
      //if (data_center == nrrdCenterCell) {
      //fh = new ScanlineField<long long>(mh, Field::EDGE);
      //} else {
      //fh = new ScanlineField<long long>(mh, Field::NODE);
      //}
      //fill_data((ScanlineField<long long>*)fh.get_rep(), n, iter, end);
      break;
    case nrrdTypeULLong :
      //if (data_center == nrrdCenterCell) {
      //fh = new ScanlineField<unsigned long long>(mh, Field::EDGE);
      //} else {
      //fh = new ScanlineField<unsigned long long>(mh, Field::NODE);
      //}
      //fill_data((ScanlineField<unsigned long long>*)fh.get_rep(), n,iter, end);
      break;
    case nrrdTypeFloat :
      if (data_center == nrrdCenterCell) {
	fh = new ScanlineField<float>(mh, Field::EDGE);
	ScanlineMesh::Edge::iterator iter, end;
	mh->begin(iter);
	mh->end(end);
	fill_data((ScanlineField<float>*)fh.get_rep(), n, iter, end);
      } else {
	fh = new ScanlineField<float>(mh, Field::NODE);
	ScanlineMesh::Node::iterator iter, end;
	mh->begin(iter);
	mh->end(end);
	fill_data((ScanlineField<float>*)fh.get_rep(), n, iter, end);
      }

      break;
    case nrrdTypeDouble :
      if (data_center == nrrdCenterCell) {
	fh = new ScanlineField<double>(mh, Field::EDGE);
	ScanlineMesh::Edge::iterator iter, end;
	mh->begin(iter);
	mh->end(end);
	fill_data((ScanlineField<double>*)fh.get_rep(), n, iter, end);
      } else {
	fh = new ScanlineField<double>(mh, Field::NODE);
	ScanlineMesh::Node::iterator iter, end;
	mh->begin(iter);
	mh->end(end);
	fill_data((ScanlineField<double>*)fh.get_rep(), n, iter, end);
      }

      break;
    }
    break;
  case 2: // Vector
    if (data_center == nrrdCenterCell) {
      fh = new ScanlineField<Vector>(mh, Field::EDGE);
      ScanlineMesh::Edge::iterator iter, end;
      mh->begin(iter);
      mh->end(end);
      fill_data((ScanlineField<Vector>*)fh.get_rep(), n, iter, end);
    } else {
      fh = new ScanlineField<Vector>(mh, Field::NODE);
      ScanlineMesh::Node::iterator iter, end;
      mh->begin(iter);
      mh->end(end);
      fill_data((ScanlineField<Vector>*)fh.get_rep(), n, iter, end);
    }

    break;
  case 6: // Tensor
    if (data_center == nrrdCenterCell) {
      fh = new ScanlineField<Tensor>(mh, Field::EDGE);
      ScanlineMesh::Edge::iterator iter, end;
      mh->begin(iter);
      mh->end(end);
      if (build_eigens_.get() && n->type == nrrdTypeFloat)
	fill_eigen_data((ScanlineField<Tensor>*)fh.get_rep(), n, iter, end);
      else
	fill_data((ScanlineField<Tensor>*)fh.get_rep(), n, iter, end);
    } else {
      fh = new ScanlineField<Tensor>(mh, Field::NODE);
      ScanlineMesh::Node::iterator iter, end;
      mh->begin(iter);
      mh->end(end);
      if (build_eigens_.get() && n->type == nrrdTypeFloat)
	fill_eigen_data((ScanlineField<Tensor>*)fh.get_rep(), n, iter, end);
      else
	fill_data((ScanlineField<Tensor>*)fh.get_rep(), n, iter, end);
    }

    break;
  default:
    cerr << "unknown index offset: " << mx_idx << endl;
    ASSERTFAIL("Unknown data size");
    break;
  }
  return fh;
}

FieldHandle 
NrrdToField::create_image_field(NrrdDataHandle &nrd) 
{
  Nrrd *n = nrd->nrrd;

  double spc[2];
  int data_center = nrrdCenterUnknown;
  
  for (int a = 1; a < 3; a++) {
    if (!(AIR_EXISTS(n->axis[a].min) && AIR_EXISTS(n->axis[a].max)))
      nrrdAxisMinMaxSet(n, a, nrrdCenterNode);
    if ( AIR_EXISTS(n->axis[a].spacing)) { spc[a-1] = n->axis[a].spacing; }
    else { spc[a-1] = 1.; }
    if (data_center == nrrdCenterUnknown) // nothing specified yet
      data_center = n->axis[a].center;
    else if (n->axis[a].center != nrrdCenterUnknown && // this one is specified
	     data_center != n->axis[a].center) { // mismatch!
      error("SCIRun cannot convert a nrrd with mismatched data centers");
      return 0;
    } // else this one was nrrdCenterUnknown, or they matched
  }

  // if nothing was specified, just call it node-centered (arbitrary)
  if (data_center == nrrdCenterUnknown) data_center = nrrdCenterNode;

  Point min(0., 0., 0.);
  Point max;
  
  if (data_center == nrrdCenterCell) {
    max = Point(n->axis[1].size * spc[0], 
		n->axis[2].size * spc[1],
		0.0);
  } else {
    max = Point((n->axis[1].size - 1) * spc[0], 
		(n->axis[2].size - 1) * spc[1],
		0.0);
  }
  int off = 0;
  if (data_center == nrrdCenterCell) { off = 1; }
  ImageMesh *m = new ImageMesh(n->axis[1].size + off, n->axis[2].size + off,
			       min, max);
  ImageMeshHandle mh(m);
  FieldHandle fh;
  int mn_idx, mx_idx;
  nrd->get_tuple_index_info(0, 0, mn_idx, mx_idx);
  
  switch (mx_idx) {
  case 0:
    switch (n->type) {
    case nrrdTypeChar :  
      if (data_center == nrrdCenterCell) {
	fh = new ImageField<char>(mh, Field::FACE);
	ImageMesh::Face::iterator iter, end;
	mh->begin(iter);
	mh->end(end);
	fill_data((ImageField<char>*)fh.get_rep(), n, iter, end);
      } else {
	fh = new ImageField<char>(mh, Field::NODE);
	ImageMesh::Node::iterator iter, end;
	mh->begin(iter);
	mh->end(end);
	fill_data((ImageField<char>*)fh.get_rep(), n, iter, end);
      }
      break;
    case nrrdTypeUChar : 
      if (data_center == nrrdCenterCell) {
	fh = new ImageField<unsigned char>(mh, Field::FACE);
	ImageMesh::Face::iterator iter, end;
	mh->begin(iter);
	mh->end(end);
	fill_data((ImageField<unsigned char>*)fh.get_rep(), n, iter, end);
      } else {
	fh = new ImageField<unsigned char>(mh, Field::NODE);
	ImageMesh::Node::iterator iter, end;
	mh->begin(iter);
	mh->end(end);
	fill_data((ImageField<unsigned char>*)fh.get_rep(), n, iter, end);
      }
      break;
    case nrrdTypeShort : 
      if (data_center == nrrdCenterCell) {
	fh = new ImageField<short>(mh, Field::FACE);
	ImageMesh::Face::iterator iter, end;
	mh->begin(iter);
	mh->end(end);
	fill_data((ImageField<short>*)fh.get_rep(), n, iter, end);
      } else {
	fh = new ImageField<short>(mh, Field::NODE);
	ImageMesh::Node::iterator iter, end;
	mh->begin(iter);
	mh->end(end);
	fill_data((ImageField<short>*)fh.get_rep(), n, iter, end);
      }
      break;
    case nrrdTypeUShort :
      if (data_center == nrrdCenterCell) {
	fh = new ImageField<unsigned short>(mh, Field::FACE);
	ImageMesh::Face::iterator iter, end;
	mh->begin(iter);
	mh->end(end);
	fill_data((ImageField<unsigned short>*)fh.get_rep(), n, iter, end);
      } else {
	fh = new ImageField<unsigned short>(mh, Field::NODE);
	ImageMesh::Node::iterator iter, end;
	mh->begin(iter);
	mh->end(end);
	fill_data((ImageField<unsigned short>*)fh.get_rep(), n, iter, end);
      }
      break;
    case nrrdTypeInt : 
      if (data_center == nrrdCenterCell) {
	fh = new ImageField<int>(mh, Field::FACE);
	ImageMesh::Face::iterator iter, end;
	mh->begin(iter);
	mh->end(end);
	fill_data((ImageField<int>*)fh.get_rep(), n, iter, end);
      } else {
	fh = new ImageField<int>(mh, Field::NODE);
	ImageMesh::Node::iterator iter, end;
	mh->begin(iter);
	mh->end(end);
	fill_data((ImageField<int>*)fh.get_rep(), n, iter, end);
      }
      break;
    case nrrdTypeUInt :  
      if (data_center == nrrdCenterCell) {
	fh = new ImageField<unsigned int>(mh, Field::FACE);
	ImageMesh::Face::iterator iter, end;
	mh->begin(iter);
	mh->end(end);
	fill_data((ImageField<unsigned int>*)fh.get_rep(), n, iter, end);
      } else {
	fh = new ImageField<unsigned int>(mh, Field::NODE);
	ImageMesh::Node::iterator iter, end;
	mh->begin(iter);
	mh->end(end);
	fill_data((ImageField<unsigned int>*)fh.get_rep(), n, iter, end);
      }
      break;
    case nrrdTypeLLong : 
      //fh = new ImageField<long long>(mh, Field::NODE);
      //fill_data((ImageField<long long>*)fh.get_rep(), n, iter, end);
      break;
    case nrrdTypeULLong :
      //fh = new ImageField<unsigned long long>(mh, Field::NODE);
      //fill_data((ImageField<unsigned long long>*)fh.get_rep(), n,iter, end);
      break;
    case nrrdTypeFloat :
      if (data_center == nrrdCenterCell) {
	fh = new ImageField<float>(mh, Field::FACE);
	ImageMesh::Face::iterator iter, end;
	mh->begin(iter);
	mh->end(end);
	fill_data((ImageField<float>*)fh.get_rep(), n, iter, end);
      } else {
	fh = new ImageField<float>(mh, Field::NODE);
	ImageMesh::Node::iterator iter, end;
	mh->begin(iter);
	mh->end(end);
	fill_data((ImageField<float>*)fh.get_rep(), n, iter, end);
      }
      break;
    case nrrdTypeDouble :
      if (data_center == nrrdCenterCell) {
	fh = new ImageField<double>(mh, Field::FACE);
	ImageMesh::Face::iterator iter, end;
	mh->begin(iter);
	mh->end(end);
	fill_data((ImageField<double>*)fh.get_rep(), n, iter, end);
      } else {
	fh = new ImageField<double>(mh, Field::NODE);
	ImageMesh::Node::iterator iter, end;
	mh->begin(iter);
	mh->end(end);
	fill_data((ImageField<double>*)fh.get_rep(), n, iter, end);
      }
      break;
    }
    break;
  case 2: // Vector
    if (data_center == nrrdCenterCell) {
      fh = new ImageField<Vector>(mh, Field::FACE);
      ImageMesh::Face::iterator iter, end;
      mh->begin(iter);
      mh->end(end);
      fill_data((ImageField<Vector>*)fh.get_rep(), n, iter, end);
    } else {
      fh = new ImageField<Vector>(mh, Field::NODE);
      ImageMesh::Node::iterator iter, end;
      mh->begin(iter);
      mh->end(end);
      fill_data((ImageField<Vector>*)fh.get_rep(), n, iter, end);
    }
    break;
  case 6: // Tensor
    if (data_center == nrrdCenterCell) {
      fh = new ImageField<Tensor>(mh, Field::FACE);
      ImageMesh::Face::iterator iter, end;
      mh->begin(iter);
      mh->end(end);
      if (build_eigens_.get() && n->type == nrrdTypeFloat)
	fill_eigen_data((ImageField<Tensor>*)fh.get_rep(), n, iter, end);
      else
	fill_data((ImageField<Tensor>*)fh.get_rep(), n, iter, end);
    } else {
      fh = new ImageField<Tensor>(mh, Field::NODE);
      ImageMesh::Node::iterator iter, end;
      mh->begin(iter);
      mh->end(end);
      if (build_eigens_.get() && n->type == nrrdTypeFloat)
	fill_eigen_data((ImageField<Tensor>*)fh.get_rep(), n, iter, end);
      else
	fill_data((ImageField<Tensor>*)fh.get_rep(), n, iter, end);
    }
    break;
  default:
    cerr << "unknown index offset: " << mx_idx << endl;
    ASSERTFAIL("Unknown data size");
    break;
  }
  return fh;
}

FieldHandle 
NrrdToField::create_latvol_field(NrrdDataHandle &nrd) 
{
  Nrrd *n = nrd->nrrd;
  double spc[3];
  int data_center = nrrdCenterUnknown;

  for (int a = 1; a < 4; a++) {
    if (!(AIR_EXISTS(n->axis[a].min) && AIR_EXISTS(n->axis[a].max)))
      nrrdAxisMinMaxSet(n, a, nrrdCenterNode);
    if ( AIR_EXISTS(n->axis[a].spacing)) { spc[a-1] = n->axis[a].spacing; }
    else { spc[a-1] = 1.; }
    if (data_center == nrrdCenterUnknown) // nothing specified yet
      data_center = n->axis[a].center;
    else if (n->axis[a].center != nrrdCenterUnknown && // this one is specified
	     data_center != n->axis[a].center) { // mismatch!
      error("SCIRun cannot convert a nrrd with mismatched data centers");
      return 0;
    } // else this one was nrrdCenterUnknown, or they matched
  }

  // if nothing was specified, just call it node-centered (arbitrary)
  if (data_center == nrrdCenterUnknown) data_center = nrrdCenterNode;

  Point min(0., 0., 0.);
  Point max;
  
  if (data_center == nrrdCenterCell) {
    max = Point(n->axis[1].size * spc[0], 
		n->axis[2].size * spc[1],
		n->axis[3].size * spc[2]);
  } else {
    max = Point((n->axis[1].size - 1) * spc[0], 
		(n->axis[2].size - 1) * spc[1],
		(n->axis[3].size - 1) * spc[2]);
  }

  int off = 0;
  if (data_center == nrrdCenterCell) { off = 1; }
  LatVolMesh *m = new LatVolMesh(n->axis[1].size + off, n->axis[2].size + off, 
				 n->axis[3].size + off, min, max);

  LatVolMeshHandle mh(m);
  FieldHandle fh;
  
  int mn_idx, mx_idx;
  nrd->get_tuple_index_info(0, 0, mn_idx, mx_idx);

  switch (mx_idx) {
  case 0:
    switch (n->type) {
    case nrrdTypeChar :  
      if (data_center == nrrdCenterCell) {
	fh = new LatVolField<char>(mh, Field::CELL);
	LatVolMesh::Cell::iterator iter, end;
	mh->begin(iter);
	mh->end(end);
	fill_data((LatVolField<char>*)fh.get_rep(), n, iter, end);
      } else {
	fh = new LatVolField<char>(mh, Field::NODE);
	LatVolMesh::Node::iterator iter, end;
	mh->begin(iter);
	mh->end(end);
	fill_data((LatVolField<char>*)fh.get_rep(), n, iter, end);
      }
      break;
    case nrrdTypeUChar : 
      if (data_center == nrrdCenterCell) {
	fh = new LatVolField<unsigned char>(mh, Field::CELL);
	LatVolMesh::Cell::iterator iter, end;
	mh->begin(iter);
	mh->end(end);
	fill_data((LatVolField<unsigned char>*)fh.get_rep(), n, iter, end);
      } else {
	fh = new LatVolField<unsigned char>(mh, Field::NODE);
	LatVolMesh::Node::iterator iter, end;
	mh->begin(iter);
	mh->end(end);
	fill_data((LatVolField<unsigned char>*)fh.get_rep(), n, iter, end);
      }
      break;
    case nrrdTypeShort : 
      if (data_center == nrrdCenterCell) {
	fh = new LatVolField<short>(mh, Field::CELL);
	LatVolMesh::Cell::iterator iter, end;
	mh->begin(iter);
	mh->end(end);
	fill_data((LatVolField<short>*)fh.get_rep(), n, iter, end);
      } else {
	fh = new LatVolField<short>(mh, Field::NODE);
	LatVolMesh::Node::iterator iter, end;
	mh->begin(iter);
	mh->end(end);
	fill_data((LatVolField<short>*)fh.get_rep(), n, iter, end);
      }
      break;
    case nrrdTypeUShort :
      if (data_center == nrrdCenterCell) {
	fh = new LatVolField<unsigned short>(mh, Field::CELL);
	LatVolMesh::Cell::iterator iter, end;
	mh->begin(iter);
	mh->end(end);
	fill_data((LatVolField<unsigned short>*)fh.get_rep(), n, iter, end);
      } else {
	fh = new LatVolField<unsigned short>(mh, Field::NODE);
	LatVolMesh::Node::iterator iter, end;
	mh->begin(iter);
	mh->end(end);
	fill_data((LatVolField<unsigned short>*)fh.get_rep(), n, iter, end);
      }
      break;
    case nrrdTypeInt : 
      if (data_center == nrrdCenterCell) {
	fh = new LatVolField<int>(mh, Field::CELL);
	LatVolMesh::Cell::iterator iter, end;
	mh->begin(iter);
	mh->end(end);
	fill_data((LatVolField<int>*)fh.get_rep(), n, iter, end);
      } else {
	fh = new LatVolField<int>(mh, Field::NODE);
	LatVolMesh::Node::iterator iter, end;
	mh->begin(iter);
	mh->end(end);
	fill_data((LatVolField<int>*)fh.get_rep(), n, iter, end);
      }
      break;
    case nrrdTypeUInt :  
      if (data_center == nrrdCenterCell) {
	fh = new LatVolField<unsigned int>(mh, Field::CELL);
	LatVolMesh::Cell::iterator iter, end;
	mh->begin(iter);
	mh->end(end);
	fill_data((LatVolField<unsigned int>*)fh.get_rep(), n, iter, end);
      } else {
	fh = new LatVolField<unsigned int>(mh, Field::NODE);
	LatVolMesh::Node::iterator iter, end;
	mh->begin(iter);
	mh->end(end);
	fill_data((LatVolField<unsigned int>*)fh.get_rep(), n, iter, end);
      }
      break;
    case nrrdTypeLLong : 
      //fh = new LatVolField<long long>(mh, Field::NODE);
      //fill_data((LatVolField<long long>*)fh.get_rep(), n, iter, end);
      break;
    case nrrdTypeULLong :
      //fh = new LatVolField<unsigned long long>(mh, Field::NODE);
      //fill_data((LatVolField<unsigned long long>*)fh.get_rep(), n,iter, end);
      break;
    case nrrdTypeFloat :
      if (data_center == nrrdCenterCell) {
	fh = new LatVolField<float>(mh, Field::CELL);
	LatVolMesh::Cell::iterator iter, end;
	mh->begin(iter);
	mh->end(end);
	fill_data((LatVolField<float>*)fh.get_rep(), n, iter, end);
      } else {
	fh = new LatVolField<float>(mh, Field::NODE);
	LatVolMesh::Node::iterator iter, end;
	mh->begin(iter);
	mh->end(end);
	fill_data((LatVolField<float>*)fh.get_rep(), n, iter, end);
      }
      break;
    case nrrdTypeDouble :
      if (data_center == nrrdCenterCell) {
	fh = new LatVolField<double>(mh, Field::CELL);
	LatVolMesh::Cell::iterator iter, end;
	mh->begin(iter);
	mh->end(end);
	fill_data((LatVolField<double>*)fh.get_rep(), n, iter, end);
      } else {
	fh = new LatVolField<double>(mh, Field::NODE);
	LatVolMesh::Node::iterator iter, end;
	mh->begin(iter);
	mh->end(end);
	fill_data((LatVolField<double>*)fh.get_rep(), n, iter, end);
      }
      break;
    }
    break;
  case 2: // Vector
    if (data_center == nrrdCenterCell) {
      fh = new LatVolField<Vector>(mh, Field::CELL);
      LatVolMesh::Cell::iterator iter, end;
      mh->begin(iter);
      mh->end(end);
      fill_data((LatVolField<Vector>*)fh.get_rep(), n, iter, end);
    } else {
      fh = new LatVolField<Vector>(mh, Field::NODE);
      LatVolMesh::Node::iterator iter, end;
      mh->begin(iter);
      mh->end(end);
      fill_data((LatVolField<Vector>*)fh.get_rep(), n, iter, end);
    }
    break;
  case 6: // Tensor
    if (data_center == nrrdCenterCell) {
      fh = new LatVolField<Tensor>(mh, Field::CELL);
      LatVolMesh::Cell::iterator iter, end;
      mh->begin(iter);
      mh->end(end);
      if (build_eigens_.get() && n->type == nrrdTypeFloat)
	fill_eigen_data((LatVolField<Tensor>*)fh.get_rep(), n, iter, end);
      else
	fill_data((LatVolField<Tensor>*)fh.get_rep(), n, iter, end);
    } else {
      fh = new LatVolField<Tensor>(mh, Field::NODE);
      LatVolMesh::Node::iterator iter, end;
      mh->begin(iter);
      mh->end(end);
      if (build_eigens_.get() && n->type == nrrdTypeFloat)
	fill_eigen_data((LatVolField<Tensor>*)fh.get_rep(), n, iter, end);
      else
	fill_data((LatVolField<Tensor>*)fh.get_rep(), n, iter, end);
    }
    break;
  default:
    cerr << "unknown index offset: " << mx_idx << endl;
    ASSERTFAIL("Unknown data size");
    break;
  }
  return fh;
}

const TypeDescription *
get_new_td(int t) {

  switch (t) {
  case nrrdTypeChar :  
    return get_type_description((char*)0);
    break;
  case nrrdTypeUChar : 
    return get_type_description((unsigned char*)0);
    break;
  case nrrdTypeShort : 
    return get_type_description((short*)0);
    break;
  case nrrdTypeUShort :
    return get_type_description((unsigned short*)0);
    break;
  case nrrdTypeInt :   
    return get_type_description((int*)0);
    break;
  case nrrdTypeUInt :  
    return get_type_description((unsigned int*)0);
    break;
  case nrrdTypeLLong : 
    //return get_type_description((long long*)0);
    break;
  case nrrdTypeULLong :
    //return get_type_description((unsigned long long*)0);
    break;
  case nrrdTypeFloat : 
    return get_type_description((float*)0);
    break;
  case nrrdTypeDouble :
    return get_type_description((double*)0);
    break;
  }
  return 0;
}

void NrrdToField::execute()
{
  NrrdDataHandle ninH;
  inrrd = (NrrdIPort *)get_iport("Nrrd");
  ofield = (FieldOPort *)get_oport("Field");

  if (!inrrd) {
    error("Unable to initialize iport 'Nrrd'.");
    return;
  }
  if (!ofield) {
    error("Unable to initialize oport 'Field'.");
    return;
  }

  if(!inrrd->get(ninH))
    return;

  Nrrd *n = ninH->nrrd;
  bool dim_based_convert = true;
  FieldHandle ofield_handle;

  if (ninH->is_sci_nrrd()) {
    // the NrrdData has a stored MeshHandle which from the originating field.
    FieldHandle fh = ninH->get_orig_field();
    const TypeDescription *td = fh->get_type_description();
    // manipilate the type to match the nrrd.
    const TypeDescription *sub = get_new_td(n->type);

    TypeDescription::td_vec *v = td->get_sub_type();
    v->clear();
    v->push_back(sub);

    CompileInfoHandle ci = ConvertToFieldBase::get_compile_info(td);
    Handle<ConvertToFieldBase> algo;
    if ((module_dynamic_compile(ci, algo)) && 
	(algo->convert_to_field(fh, ninH, ofield_handle))) 
    {
      remark("Creating a Field from original mesh in input nrrd");
      dim_based_convert = false;
    }
    // if compilation fails or the algo cant match the data to the mesh,
    // do a standard dimemsion based convert.
  }

  if (dim_based_convert) {

    int dim = n->dim;
    // have always dim + 1 axes 
    --dim;

    switch (dim) {

    case 1:
      {
	//get data from x axis and stuff into a Scanline
	remark("Creating a ScanlineField from input nrrd");
	ofield_handle = create_scanline_field(ninH);
      }
      break;
    case 2:
      {
	//get data from x,y axes and stuff into an Image
	remark("Creating a ImageField from input nrrd");
	ofield_handle = create_image_field(ninH);
      }
      break;
    case 3:
      {
	//get data from x,y,z axes and stuff into a LatVol
	remark("Creating a LatVolField from input nrrd");
	ofield_handle = create_latvol_field(ninH);
      }
      break;
    default:
      error("Cannot convert > 3 dimesional data to a SCIRun Field.");
      return;
    }
  }
 
  ofield->send(ofield_handle);
}
