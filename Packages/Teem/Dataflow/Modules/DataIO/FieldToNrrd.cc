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
  if (fieldH->get_type_name(0) != "LatticeVol") {
    cerr << "Error FieldToNrrd only works with LatticeVol's" << endl;
    return;
  }

  int nx, ny, nz;
  NrrdData *nout=scinew NrrdData;
  Field *field = fieldH.get_rep();
  const string data = field->get_type_name(1);
  if (field->data_at() != Field::NODE) {
    cerr << "Error - can only build a nrrd from data at nodes.\n";
    return;
  }
  LatVolMeshHandle lvm;
  nout->nrrd = nrrdNew();
  if (data == "double") { 
    LatticeVol<double> *f = 
      dynamic_cast<LatticeVol<double>*>(field);
    lvm = f->get_typed_mesh();
    nx = lvm->get_nx();
    ny = lvm->get_ny();
    nz = lvm->get_nz();
    double *data=new double[nx*ny*nz];
    double *p=&(data[0]);
    for (int k=0; k<nz; k++)
      for (int j=0; j<ny; j++)
	for (int i=0; i<nx; i++)
	  *p++=f->fdata()(k,j,i);
    nrrdWrap(nout->nrrd, data, nx*ny*nz, nrrdTypeDouble, 3);
  } else if (data == "float") { 
    LatticeVol<float> *f = 
      dynamic_cast<LatticeVol<float>*>(field);
    lvm = f->get_typed_mesh();
    nx = lvm->get_nx();
    ny = lvm->get_ny();
    nz = lvm->get_nz();
    float *data=new float[nx*ny*nz];
    float *p=&(data[0]);
    for (int k=0; k<nz; k++)
      for (int j=0; j<ny; j++)
	for (int i=0; i<nx; i++)
	  *p++=f->fdata()(k,j,i);
    nrrdWrap(nout->nrrd, data, nx*ny*nz, nrrdTypeFloat, 3);
#if 0
  } else if (data == "unsigned_int") {
    LatticeVol<unsigned int> *f = 
      dynamic_cast<LatticeVol<unsigned int>*>(field);
    lvm = f->get_typed_mesh();
    nx = lvm->get_nx();
    ny = lvm->get_ny();
    nz = lvm->get_nz();
    unsigned int *data=new unsigned int[nx*ny*nz];
    unsigned int *p=&(data[0]);
    for (int k=0; k<nz; k++)
      for (int j=0; j<ny; j++)
	for (int i=0; i<nx; i++)
	  *p++=f->fdata()(k,j,i);
    nrrdWrap(nout->nrrd, data, nx*ny*nz, nrrdTypeUInt, 3);
#endif
  } else if (data == "int") {
    LatticeVol<int> *f = 
      dynamic_cast<LatticeVol<int>*>(field);
    lvm = f->get_typed_mesh();
    nx = lvm->get_nx();
    ny = lvm->get_ny();
    nz = lvm->get_nz();
    int *data=new int[nx*ny*nz];
    int *p=&(data[0]);
    for (int k=0; k<nz; k++)
      for (int j=0; j<ny; j++)
	for (int i=0; i<nx; i++)
	  *p++=f->fdata()(k,j,i);
    nrrdWrap(nout->nrrd, data, nx*ny*nz, nrrdTypeInt, 3);
#if 0
  } else if (data == "unsigned_short") {
    LatticeVol<unsigned short> *f = 
      dynamic_cast<LatticeVol<unsigned short>*>(field);
    lvm = f->get_typed_mesh();
    nx = lvm->get_nx();
    ny = lvm->get_ny();
    nz = lvm->get_nz();
    unsigned short *data=new unsigned short[nx*ny*nz];
    unsigned short *p=&(data[0]);
    for (int k=0; k<nz; k++)
      for (int j=0; j<ny; j++)
	for (int i=0; i<nx; i++)
	  *p++=f->fdata()(k,j,i);
    nrrdWrap(nout->nrrd, data, nx*ny*nz, nrrdTypeUShort, 3);
#endif
  } else if (data == "short") {
    LatticeVol<short> *f = 
      dynamic_cast<LatticeVol<short>*>(field);
    lvm = f->get_typed_mesh();
    nx = lvm->get_nx();
    ny = lvm->get_ny();
    nz = lvm->get_nz();
    short *data=new short[nx*ny*nz];
    short *p=&(data[0]);
    for (int k=0; k<nz; k++)
      for (int j=0; j<ny; j++)
	for (int i=0; i<nx; i++)
	  *p++=f->fdata()(k,j,i);
    nrrdWrap(nout->nrrd, data, nx*ny*nz, nrrdTypeShort, 3);
  } else if (data == "unsigned_char") {
    LatticeVol<unsigned char> *f = 
      dynamic_cast<LatticeVol<unsigned char>*>(field);
    lvm = f->get_typed_mesh();
    nx = lvm->get_nx();
    ny = lvm->get_ny();
    nz = lvm->get_nz();
    unsigned char *data=new unsigned char[nx*ny*nz];
    unsigned char *p=&(data[0]);
    for (int k=0; k<nz; k++)
      for (int j=0; j<ny; j++)
	for (int i=0; i<nx; i++)
	  *p++=f->fdata()(k,j,i);
    nrrdWrap(nout->nrrd, data, nx*ny*nz, nrrdTypeUChar, 3);
#if 0
  } else if (data == "char") {
    LatticeVol<char> *f = 
      dynamic_cast<LatticeVol<char>*>(field);
    lvm = f->get_typed_mesh();
    nx = lvm->get_nx();
    ny = lvm->get_ny();
    nz = lvm->get_nz();
    char *data=new char[nx*ny*nz];
    char *p=&(data[0]);
    for (int k=0; k<nz; k++)
      for (int j=0; j<ny; j++)
	for (int i=0; i<nx; i++)
	  *p++=f->fdata()(k,j,i);
    nrrdWrap(nout->nrrd, data, nx*ny*nz, nrrdTypeChar, 3);
#endif
  } else if (data == "Vector") {
    LatticeVol<Vector> *f = 
      dynamic_cast<LatticeVol<Vector>*>(field);
    lvm = f->get_typed_mesh();
    nx = lvm->get_nx();
    ny = lvm->get_ny();
    nz = lvm->get_nz();
    double *data=new double[nx*ny*nz*3];
    double *p=&(data[0]);
    for (int k=0; k<nz; k++)
      for (int j=0; j<ny; j++)
	for (int i=0; i<nx; i++) {
	  *p++=f->fdata()(k,j,i).x();
	  *p++=f->fdata()(k,j,i).y();
	  *p++=f->fdata()(k,j,i).z();
	}
    nrrdWrap(nout->nrrd, data, nx*ny*nz*3, nrrdTypeDouble, 4);
  } else if (data == "Tensor") {
    LatticeVol<Tensor> *f = 
      dynamic_cast<LatticeVol<Tensor>*>(field);
    lvm = f->get_typed_mesh();
    nx = lvm->get_nx();
    ny = lvm->get_ny();
    nz = lvm->get_nz();
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
    nrrdWrap(nout->nrrd, data, nx*ny*nz*7, nrrdTypeDouble, 7);
  } else {
    cerr << "Error - unknown LatticeVol data type " << data << endl;
    free(nout);
    return;
  }

  Point minP = lvm->get_min();
  Point maxP = lvm->get_max();

  ScalarFieldInterface *sfi = field->query_scalar_interface();
  if (sfi)
  {
    double minv, maxv;
    sfi->compute_min_max(minv, maxv);
    nout->nrrd->min = minv;
    nout->nrrd->max = maxv;
  }
  Vector v(maxP-minP);
  nout->nrrd->axis[0].size=nx;
  nout->nrrd->axis[1].size=ny;
  nout->nrrd->axis[2].size=nz;
  v.x(v.x()/(nx-1));
  v.y(v.y()/(ny-1));
  v.z(v.z()/(nz-1));
  nout->nrrd->axis[0].min=minP.x();
  nout->nrrd->axis[1].min=minP.y();
  nout->nrrd->axis[2].min=minP.z();
  nout->nrrd->axis[0].max=maxP.x();
  nout->nrrd->axis[1].max=maxP.y();
  nout->nrrd->axis[2].max=maxP.z();
  nout->nrrd->axis[0].spacing=v.x();
  nout->nrrd->axis[1].spacing=v.y();
  nout->nrrd->axis[2].spacing=v.z();
  nout->nrrd->axis[0].label = strdup("x");
  nout->nrrd->axis[1].label = strdup("y");
  nout->nrrd->axis[2].label = strdup("z");
  if (data == "Vector") {
    nout->nrrd->axis[3].size=3;
    nout->nrrd->axis[3].min=0;
    nout->nrrd->axis[3].max=3;
    nout->nrrd->axis[3].label = strdup("v");
  } else if (data == "Tensor") {
    nout->nrrd->axis[3].size=6;
    nout->nrrd->axis[3].min=0;
    nout->nrrd->axis[3].max=6;
    nout->nrrd->axis[3].label = strdup("t");
  }

  NrrdDataHandle noutH(nout);
  onrrd->send(noutH);
}

} // End namespace SCITeem
