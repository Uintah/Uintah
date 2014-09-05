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
#include <Core/Datatypes/LatticeVol.h>
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
  FieldToNrrd(const clString& id);
  virtual ~FieldToNrrd();
  virtual void execute();
};

extern "C" Module* make_FieldToNrrd(const clString& id) {
  return new FieldToNrrd(id);
}


FieldToNrrd::FieldToNrrd(const clString& id):Module("FieldToNrrd", id, Filter)
{
  // Create the input port
  ifield = scinew FieldIPort(this, "Field", FieldIPort::Atomic);
  add_iport(ifield);
  
  // Create the output ports
  onrrd = scinew NrrdOPort(this, "Nrrd", NrrdIPort::Atomic);
  add_oport(onrrd);
}

FieldToNrrd::~FieldToNrrd()
{
}
  
void FieldToNrrd::execute()
{
  FieldHandle fieldH;
  if (!ifield->get(fieldH))
    return;
  if (fieldH->get_type_name(0) != "LatticeVol") {
    cerr << "Error FieldToNrrd only works with LatticeVol's" << endl;
    return;
  }

  int nx, ny, nz;
  pair<double,double> minmax;

  NrrdData *nout=scinew NrrdData;
  Field *field = fieldH.get_rep();
  const string data = field->get_type_name(1);
  if (field->data_at() != Field::NODE) {
    cerr << "Error - can only build a nrrd from data at nodes.\n";
    return;
  }
  LatVolMeshHandle lvm;
  if (data == "double") { 
    LatticeVol<double> *f = 
      dynamic_cast<LatticeVol<double>*>(field);
    lvm = f->get_typed_mesh();
    nx = lvm->get_nx();
    ny = lvm->get_ny();
    nz = lvm->get_nz();
    field_minmax(*f, minmax);
    double *data=new double[nx*ny*nz];
    double *p=&(data[0]);
    for (int i=0; i<nx; i++)
      for (int j=0; j<ny; j++)
	for (int k=0; k<nz; k++)
	  *p++=f->fdata()(i,j,k);
    nout->nrrd=nrrdNewWrap(data, nx*ny*nz, nrrdTypeDouble, 3);
  } else if (data == "float") { 
    LatticeVol<float> *f = 
      dynamic_cast<LatticeVol<float>*>(field);
    lvm = f->get_typed_mesh();
    nx = lvm->get_nx();
    ny = lvm->get_ny();
    nz = lvm->get_nz();
    field_minmax(*f, minmax);
    float *data=new float[nx*ny*nz];
    float *p=&(data[0]);
    for (int i=0; i<nx; i++)
      for (int j=0; j<ny; j++)
	for (int k=0; k<nz; k++)
	  *p++=f->fdata()(i,j,k);
    nout->nrrd=nrrdNewWrap(data, nx*ny*nz, nrrdTypeFloat, 3);
  } else if (data == "unsigned int") {
    LatticeVol<unsigned int> *f = 
      dynamic_cast<LatticeVol<unsigned int>*>(field);
    lvm = f->get_typed_mesh();
    nx = lvm->get_nx();
    ny = lvm->get_ny();
    nz = lvm->get_nz();
    field_minmax(*f, minmax);
    unsigned int *data=new unsigned int[nx*ny*nz];
    unsigned int *p=&(data[0]);
    for (int i=0; i<nx; i++)
      for (int j=0; j<ny; j++)
	for (int k=0; k<nz; k++)
	  *p++=f->fdata()(i,j,k);
    nout->nrrd=nrrdNewWrap(data, nx*ny*nz, nrrdTypeUInt, 3);
  } else if (data == "int") {
    LatticeVol<int> *f = 
      dynamic_cast<LatticeVol<int>*>(field);
    lvm = f->get_typed_mesh();
    nx = lvm->get_nx();
    ny = lvm->get_ny();
    nz = lvm->get_nz();
    field_minmax(*f, minmax);
    int *data=new int[nx*ny*nz];
    int *p=&(data[0]);
    for (int i=0; i<nx; i++)
      for (int j=0; j<ny; j++)
	for (int k=0; k<nz; k++)
	  *p++=f->fdata()(i,j,k);
    nout->nrrd=nrrdNewWrap(data, nx*ny*nz, nrrdTypeInt, 3);
  } else if (data == "unsigned short") {
    LatticeVol<unsigned short> *f = 
      dynamic_cast<LatticeVol<unsigned short>*>(field);
    lvm = f->get_typed_mesh();
    nx = lvm->get_nx();
    ny = lvm->get_ny();
    nz = lvm->get_nz();
    field_minmax(*f, minmax);
    unsigned short *data=new unsigned short[nx*ny*nz];
    unsigned short *p=&(data[0]);
    for (int i=0; i<nx; i++)
      for (int j=0; j<ny; j++)
	for (int k=0; k<nz; k++)
	  *p++=f->fdata()(i,j,k);
    nout->nrrd=nrrdNewWrap(data, nx*ny*nz, nrrdTypeUShort, 3);
  } else if (data == "short") {
    LatticeVol<short> *f = 
      dynamic_cast<LatticeVol<short>*>(field);
    lvm = f->get_typed_mesh();
    nx = lvm->get_nx();
    ny = lvm->get_ny();
    nz = lvm->get_nz();
    field_minmax(*f, minmax);
    short *data=new short[nx*ny*nz];
    short *p=&(data[0]);
    for (int i=0; i<nx; i++)
      for (int j=0; j<ny; j++)
	for (int k=0; k<nz; k++)
	  *p++=f->fdata()(i,j,k);
    nout->nrrd=nrrdNewWrap(data, nx*ny*nz, nrrdTypeShort, 3);
  } else if (data == "unsigned char") {
    LatticeVol<unsigned char> *f = 
      dynamic_cast<LatticeVol<unsigned char>*>(field);
    lvm = f->get_typed_mesh();
    nx = lvm->get_nx();
    ny = lvm->get_ny();
    nz = lvm->get_nz();
    field_minmax(*f, minmax);
    unsigned char *data=new unsigned char[nx*ny*nz];
    unsigned char *p=&(data[0]);
    for (int i=0; i<nx; i++)
      for (int j=0; j<ny; j++)
	for (int k=0; k<nz; k++)
	  *p++=f->fdata()(i,j,k);
    nout->nrrd=nrrdNewWrap(data, nx*ny*nz, nrrdTypeUChar, 3);
  } else if (data == "char") {
    LatticeVol<char> *f = 
      dynamic_cast<LatticeVol<char>*>(field);
    lvm = f->get_typed_mesh();
    nx = lvm->get_nx();
    ny = lvm->get_ny();
    nz = lvm->get_nz();
    field_minmax(*f, minmax);
    char *data=new char[nx*ny*nz];
    char *p=&(data[0]);
    for (int i=0; i<nx; i++)
      for (int j=0; j<ny; j++)
	for (int k=0; k<nz; k++)
	  *p++=f->fdata()(i,j,k);
    nout->nrrd=nrrdNewWrap(data, nx*ny*nz, nrrdTypeChar, 3);
  } else if (data == "Vector") {
    LatticeVol<Vector> *f = 
      dynamic_cast<LatticeVol<Vector>*>(field);
    lvm = f->get_typed_mesh();
    nx = lvm->get_nx();
    ny = lvm->get_ny();
    nz = lvm->get_nz();
    double *data=new double[nx*ny*nz*3];
    double *p=&(data[0]);
    for (int i=0; i<nx; i++)
      for (int j=0; j<ny; j++)
	for (int k=0; k<nz; k++) {
	  *p++=f->fdata()(i,j,k).x();
	  *p++=f->fdata()(i,j,k).y();
	  *p++=f->fdata()(i,j,k).z();
	}
    nout->nrrd=nrrdNewWrap(data, nx*ny*nz*3, nrrdTypeDouble, 4);
  } else if (data == "Tensor") {
    LatticeVol<Tensor> *f = 
      dynamic_cast<LatticeVol<Tensor>*>(field);
    lvm = f->get_typed_mesh();
    nx = lvm->get_nx();
    ny = lvm->get_ny();
    nz = lvm->get_nz();
    double *data=new double[nx*ny*nz*7];
    double *p=&(data[0]);
    for (int i=0; i<nx; i++)
      for (int j=0; j<ny; j++)
	for (int k=0; k<nz; k++) {
	  *p++=1;  // should use mask if present
	  *p++=f->fdata()(i,j,k).mat_[0][0];
	  *p++=f->fdata()(i,j,k).mat_[0][1];
	  *p++=f->fdata()(i,j,k).mat_[0][2];
	  *p++=f->fdata()(i,j,k).mat_[1][1];
	  *p++=f->fdata()(i,j,k).mat_[1][2];
	  *p++=f->fdata()(i,j,k).mat_[2][2];
	}
    nout->nrrd=nrrdNewWrap(data, nx*ny*nz*3, nrrdTypeDouble, 4);
  } else {
    cerr << "Error - unknown LatticeVol data type " << data << endl;
    free(nout);
    return;
  }

  Point minP = lvm->get_min();
  Point maxP = lvm->get_max();
  if (data != "Vector" && data != "Tensor") {
    nout->nrrd->min=minmax.first;
    nout->nrrd->max=minmax.second;
  }
  nout->nrrd->encoding=nrrdEncodingRaw;
  nout->nrrd->size[0]=nx;
  nout->nrrd->size[1]=ny;
  nout->nrrd->size[2]=nz;
  nout->nrrd->axisMin[0]=minP.x();
  nout->nrrd->axisMin[1]=minP.y();
  nout->nrrd->axisMin[2]=minP.z();
  nout->nrrd->axisMax[0]=maxP.x();
  nout->nrrd->axisMax[1]=maxP.y();
  nout->nrrd->axisMax[2]=maxP.z();
  nout->nrrd->label[0][0]='x';
  nout->nrrd->label[0][1]='\0';
  nout->nrrd->label[1][0]='y';
  nout->nrrd->label[1][1]='\0';
  nout->nrrd->label[2][0]='z';
  nout->nrrd->label[2][1]='\0';
  if (data == "Vector") {
    nout->nrrd->size[3]=3;
    nout->nrrd->axisMin[3]=0;
    nout->nrrd->axisMax[3]=3;
    nout->nrrd->label[3][0]='v';
    nout->nrrd->label[3][1]='\0';
  } else if (data == "Tensor") {
    nout->nrrd->size[3]=6;
    nout->nrrd->axisMin[3]=0;
    nout->nrrd->axisMax[3]=6;
    nout->nrrd->label[3][0]='t';
    nout->nrrd->label[3][1]='\0';
  }

  NrrdDataHandle noutH(nout);
  onrrd->send(noutH);
}

} // End namespace SCITeem
