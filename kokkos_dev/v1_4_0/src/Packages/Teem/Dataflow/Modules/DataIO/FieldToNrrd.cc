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
 *  FieldToNrrd.cc:  Convert a Nrrd to a Field
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
#include <Teem/Dataflow/Ports/NrrdPort.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Datatypes/LatVolField.h>
#include <Core/Malloc/Allocator.h>
#include <iostream>
#include <pair.h>

using std::cerr;
using std::endl;
using std::pair;

namespace SCITeem {

using namespace SCIRun;

class FieldToNrrd : public Module {
  FieldIPort* ifield;
  NrrdOPort* onrrd;
public:
  FieldToNrrd(const string& id);
  virtual ~FieldToNrrd();
  virtual void execute();
};

extern "C" Module* make_FieldToNrrd(const string& id) {
  return new FieldToNrrd(id);
}


FieldToNrrd::FieldToNrrd(const string& id):Module("FieldToNrrd", id, Filter, "DataIO", "Teem")
{
}

FieldToNrrd::~FieldToNrrd()
{
}

#define COPY_INTO_NRRD_FROM_FIELD(type, Type) \
    LatVolField<type> *f = \
      dynamic_cast<LatVolField<type>*>(field); \
    lvm = f->get_typed_mesh(); \
    nx = f->fdata().dim3(); \
    ny = f->fdata().dim2(); \
    nz = f->fdata().dim1(); \
    type *data=new type[nx*ny*nz]; \
    type *p=&(data[0]); \
    for (int k=0; k<nz; k++) \
      for (int j=0; j<ny; j++) \
	for (int i=0; i<nx; i++) \
	  *p++=f->fdata()(k,j,i); \
    nrrdWrap(nout->nrrd, data, nrrdType##Type, 3, nx, ny, nz); \
    if (f->data_at() == Field::NODE) \
      nrrdAxesSet(nout->nrrd, nrrdAxesInfoCenter, \
                  nrrdCenterNode, nrrdCenterNode, nrrdCenterNode); \
    else // if (f->data_at() == Field::CELL) \
      nrrdAxesSet(nout->nrrd, nrrdAxesInfoCenter, \
                  nrrdCenterCell, nrrdCenterCell, nrrdCenterCell)

void FieldToNrrd::execute()
{
  ifield = (FieldIPort *)get_iport("Field");
  onrrd = (NrrdOPort *)get_oport("Nrrd");

  if (!ifield) {
    postMessage("Unable to initialize "+name+"'s iport\n");
    return;
  }
  if (!onrrd) {
    postMessage("Unable to initialize "+name+"'s oport\n");
    return;
  }

  FieldHandle fieldH;
  if (!ifield->get(fieldH))
    return;
  if (fieldH->get_type_name(0) != "LatVolField") {
    cerr << "Error FieldToNrrd only works with LatVolField's" << endl;
    return;
  }

  int nx, ny, nz;
  NrrdData *nout=scinew NrrdData;
  Field *field = fieldH.get_rep();
  const string data = field->get_type_name(1);

  nout->nrrd = nrrdNew();
  LatVolMeshHandle lvm;

  if (data =="double") {
    COPY_INTO_NRRD_FROM_FIELD(double, Double);
  } else if (data == "float") { 
    COPY_INTO_NRRD_FROM_FIELD(float, Float);
  } else if (data == "unsigned_int") {
    COPY_INTO_NRRD_FROM_FIELD(unsigned int, UInt);
  } else if (data == "int") {
    COPY_INTO_NRRD_FROM_FIELD(int, Int);
  } else if (data == "unsigned_short") {
    COPY_INTO_NRRD_FROM_FIELD(unsigned short, UShort);
  } else if (data == "short") {
    COPY_INTO_NRRD_FROM_FIELD(short, Short);
  } else if (data == "unsigned_char") {
    COPY_INTO_NRRD_FROM_FIELD(unsigned char, UChar);
  } else if (data == "char") {
    COPY_INTO_NRRD_FROM_FIELD(char, Char);
  } else if (data == "Vector") {
    LatVolField<Vector> *f = 
      dynamic_cast<LatVolField<Vector>*>(field);
    nx = f->fdata().dim3();
    ny = f->fdata().dim2();
    nz = f->fdata().dim1();
    double *data=new double[nx*ny*nz*3];
    double *p=&(data[0]);
    for (int k=0; k<nz; k++)
      for (int j=0; j<ny; j++)
	for (int i=0; i<nx; i++) {
	  *p++=f->fdata()(k,j,i).x();
	  *p++=f->fdata()(k,j,i).y();
	  *p++=f->fdata()(k,j,i).z();
	}
    nrrdWrap(nout->nrrd, data, nrrdTypeDouble, 4, 3, nx, ny, nz);
    if (f->data_at() == Field::NODE)
      nrrdAxesSet(nout->nrrd, nrrdAxesInfoCenter, nrrdCenterNode, 
		  nrrdCenterNode, nrrdCenterNode, nrrdCenterNode);
    else // if (f->data_at() == Field::CELL)
      nrrdAxesSet(nout->nrrd, nrrdAxesInfoCenter, nrrdCenterCell, 
		  nrrdCenterCell, nrrdCenterCell, nrrdCenterCell);
  } else if (data == "Tensor") {
    LatVolField<Tensor> *f = 
      dynamic_cast<LatVolField<Tensor>*>(field);
    nx = f->fdata().dim3();
    ny = f->fdata().dim2();
    nz = f->fdata().dim1();
    double *data=new double[nx*ny*nz*7];
    double *p=&(data[0]);
    for (int k=0; k<nz; k++)
      for (int j=0; j<ny; j++)
	for (int i=0; i<nx; i++) {
	  *p++=1;  // should use mask if present
	  *p++=f->fdata()(k,j,i).mat_[0][0];
	  *p++=f->fdata()(k,j,i).mat_[0][1];
	  *p++=f->fdata()(k,j,i).mat_[0][2];
	  *p++=f->fdata()(k,j,i).mat_[1][1];
	  *p++=f->fdata()(k,j,i).mat_[1][2];
	  *p++=f->fdata()(k,j,i).mat_[2][2];
	}
    nrrdWrap(nout->nrrd, data, nrrdTypeDouble, 4, 7, nx, ny, nz);
    if (f->data_at() == Field::NODE)
      nrrdAxesSet(nout->nrrd, nrrdAxesInfoCenter, nrrdCenterNode, 
		  nrrdCenterNode, nrrdCenterNode, nrrdCenterNode);
    else // if (f->data_at() == Field::CELL)
      nrrdAxesSet(nout->nrrd, nrrdAxesInfoCenter, nrrdCenterCell, 
		  nrrdCenterCell, nrrdCenterCell, nrrdCenterCell);
  } else {
    cerr << "Error - unknown LatVolField data type " << data << endl;
    free(nout);
    return;
  }

  {
    // Check for simple scale/translate matrix.
    double td[16];
    lvm->get_transform().get(td);
    if ((td[1] + td[2] + td[4] + td[6] + td[8] + td[9]) > 1.0e-3)
    {
      cerr << "NOT A STRICTLY SCALE/TRANSLATE MATRIX, WILL NOT WORK.\n";
    }
  }

  BBox bbox = lvm->get_bounding_box();
  Point minP = bbox.min();
  Point maxP = bbox.max();

  ScalarFieldInterface *sfi = field->query_scalar_interface();
  if (sfi)
  {
    double minv, maxv;
    sfi->compute_min_max(minv, maxv);
    nout->nrrd->min = minv;
    nout->nrrd->max = maxv;
  }
  Vector v(maxP-minP);
  v.x(v.x()/(nx-1));
  v.y(v.y()/(ny-1));
  v.z(v.z()/(nz-1));
  int offset=0;
  if (data == "Vector" || data == "Tensor") offset=1;
  nout->nrrd->axis[0+offset].min=minP.x();
  nout->nrrd->axis[1+offset].min=minP.y();
  nout->nrrd->axis[2+offset].min=minP.z();
  nout->nrrd->axis[0+offset].max=maxP.x();
  nout->nrrd->axis[1+offset].max=maxP.y();
  nout->nrrd->axis[2+offset].max=maxP.z();
  nout->nrrd->axis[0+offset].spacing=v.x();
  nout->nrrd->axis[1+offset].spacing=v.y();
  nout->nrrd->axis[2+offset].spacing=v.z();
  nout->nrrd->axis[0+offset].label = strdup("x");
  nout->nrrd->axis[1+offset].label = strdup("y");
  nout->nrrd->axis[2+offset].label = strdup("z");
  if (data == "Vector") {
    nout->nrrd->axis[0].label = strdup("v");
  } else if (data == "Tensor") {
    nout->nrrd->axis[0].label = strdup("t");
  }

  NrrdDataHandle noutH(nout);
  onrrd->send(noutH);
}

} // End namespace SCITeem
