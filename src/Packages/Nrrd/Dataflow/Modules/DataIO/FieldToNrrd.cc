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
#include <Nrrd/Dataflow/Ports/NrrdPort.h>
#include <Dataflow/Ports/FieldPort.h>
//#include <Core/Datatypes/LatticeVol.h>
#include <Core/Malloc/Allocator.h>
#include <iostream>

using std::cerr;
using std::endl;

namespace SCINrrd {

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
  FieldHandle sfH;
  if (!ifield->get(sfH))
    return;

  NrrdData* nout=scinew NrrdData;
  int nx, ny, nz;

// waiting for Fields...
#if 0
  ScalarFieldRGBase* sfb=sfH->getRGBase();
  if (!sfb) {
    cerr << "FieldToNrrd error - scalarfield wasn't regular.\n";
    return;
  }

  nx=sfb->nx;
  ny=sfb->ny;
  nz=sfb->nz;

  ScalarFieldRGdouble *ifd=dynamic_cast<ScalarFieldRGdouble*>(sfb);
  ScalarFieldRGfloat *iff=dynamic_cast<ScalarFieldRGfloat*>(sfb);
  ScalarFieldRGint *ifi=dynamic_cast<ScalarFieldRGint*>(sfb);
  ScalarFieldRGshort *ifs=dynamic_cast<ScalarFieldRGshort*>(sfb);
  ScalarFieldRGushort *ifus=dynamic_cast<ScalarFieldRGushort*>(sfb);
  ScalarFieldRGchar *ifc=dynamic_cast<ScalarFieldRGchar*>(sfb);
  ScalarFieldRGuchar *ifu=dynamic_cast<ScalarFieldRGuchar*>(sfb);

  if (ifu) {
    uchar *data=new uchar[nx*ny*nz];
    uchar *p=&(data[0]);
    for (int i=0; i<nx; i++)
      for (int j=0; j<ny; j++)
	for (int k=0; k<nz; k++)
	  *p++=ifu->grid(i,j,k);
    nout->nrrd=nrrdNewWrap(data, nx*ny*nz, nrrdTypeUChar, 3);
  } else if (ifc) {
    char *data=new char[nx*ny*nz];
    char *p=&(data[0]);
    for (int k=0; k<nz; k++)
      for (int j=0; j<ny; j++)
	for (int i=0; i<nx; i++)
	  *p++=ifc->grid(i,j,k);
    nout->nrrd=nrrdNewWrap(data, nx*ny*nz, nrrdTypeChar, 3);
  } else if (ifus) {
    unsigned short *data=new unsigned short[nx*ny*nz];
    unsigned short *p=&(data[0]);
    for (int k=0; k<nz; k++)
      for (int j=0; j<ny; j++)
	for (int i=0; i<nx; i++)
	  *p++=ifs->grid(i,j,k);
    nout->nrrd=nrrdNewWrap(data, nx*ny*nz, nrrdTypeUShort, 3);
  } else if (ifs) {
    short *data=new short[nx*ny*nz];
    short *p=&(data[0]);
    for (int k=0; k<nz; k++)
      for (int j=0; j<ny; j++)
	for (int i=0; i<nx; i++)
	  *p++=ifs->grid(i,j,k);
    nout->nrrd=nrrdNewWrap(data, nx*ny*nz, nrrdTypeShort, 3);
  } else if (ifi) {
    int *data=new int[nx*ny*nz];
    int *p=&(data[0]);
    for (int k=0; k<nz; k++)
      for (int j=0; j<ny; j++)
	for (int i=0; i<nx; i++)
	  *p++=ifi->grid(i,j,k);
    nout->nrrd=nrrdNewWrap(data, nx*ny*nz, nrrdTypeInt, 3);
  } else if (iff) {
    float *data=new float[nx*ny*nz];
    float *p=&(data[0]);
    for (int k=0; k<nz; k++)
      for (int j=0; j<ny; j++)
	for (int i=0; i<nx; i++)
	  *p++=iff->grid(i,j,k);
    nout->nrrd=nrrdNewWrap(data, nx*ny*nz, nrrdTypeFloat, 3);
  } else if (ifs) {
    double *data=new double[nx*ny*nz];
    double *p=&(data[0]);
    for (int k=0; k<nz; k++)
      for (int j=0; j<ny; j++)
	for (int i=0; i<nx; i++)
	  *p++=ifd->grid(i,j,k);
    nout->nrrd=nrrdNewWrap(data, nx*ny*nz, nrrdTypeDouble, 3);
  } else {
    cerr << "FieldToNrrd error - unrecognized scalarfieldrgbase type.\n";
    return;
  }
#endif  

  double min, max;
//  sfb->get_minmax(min, max);
  nout->nrrd->min=min;
  nout->nrrd->max=max;
  nout->nrrd->encoding=nrrdEncodingRaw;
  nout->nrrd->size[0]=nx;
  nout->nrrd->size[1]=ny;
  nout->nrrd->size[2]=nz;
  Point minP, maxP;
//  sfb->get_bounds(minP, maxP);
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

  NrrdDataHandle noutH(nout);
  onrrd->send(noutH);
}

} // End namespace SCINrrd
