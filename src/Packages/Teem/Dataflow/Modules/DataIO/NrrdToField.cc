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
#include <Teem/Dataflow/Ports/NrrdPort.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Datatypes/MaskedLatticeVol.h>
#include <Core/Malloc/Allocator.h>
#include <iostream>

using std::cerr;
using std::endl;

namespace SCITeem {

using namespace SCIRun;

class NrrdToField : public Module {
  NrrdIPort* inrrd;
  FieldOPort* ofield;
public:
  NrrdToField(const string& id);
  virtual ~NrrdToField();
  virtual void execute();
};

extern "C" Module* make_NrrdToField(const string& id) {
  return new NrrdToField(id);
}


NrrdToField::NrrdToField(const string& id):Module("NrrdToField", id, Filter)
{
  // Create the input port
  inrrd = scinew NrrdIPort(this, "Nrrd", NrrdIPort::Atomic);
  add_iport(inrrd);
  
  // Create the output ports
  ofield = scinew FieldOPort(this, "Field", FieldIPort::Atomic);
  add_oport(ofield);
}

NrrdToField::~NrrdToField()
{
}

void NrrdToField::execute()
{
  NrrdDataHandle ninH;
  if(!inrrd->get(ninH))
    return;

  Nrrd *n = ninH->nrrd;
  int i, j, k;

  FieldHandle fieldH;

  if (n->dim == 4) {  // vector or tensor data
    if (n->type != nrrdTypeFloat) {
      cerr << "Error - tensor nrrd's must be floats.\n";
      return;
    }
    // matrix, x, y, z
    if (n->size[0] == 7) {
      // the Nrrd assumed samples at nodes and gave min/max accordingly
      // but we want to think of those samples as centers of cells, so
      // we need a different mesh
      Point minP(0,0,0);
      Point maxP((n->size[1]+1)*n->spacing[1], (n->size[2]+1)*n->spacing[2],
		 (n->size[3]+1)*n->spacing[3]);
      int nx = n->size[1];
      int ny = n->size[2];
      int nz = n->size[3];
      LatVolMesh *lvm = scinew LatVolMesh(nx+1, ny+1, nz+1, minP, maxP);
      MaskedLatticeVol<Tensor> *f =
	scinew MaskedLatticeVol<Tensor>(lvm, Field::CELL);
      double *p=(double *)n->data;
      Array1<double> tens(6);
      for (k=0; k<nz; k++)
	for (j=0; j<ny; j++)
	  for (i=0; i<nx; i++, p++) {
	    if (*p++ > .5) f->mask()(k,j,i)=1; else f->mask()(k,j,i)=0;
	    for (int ch=0; ch<6; ch++) tens[ch]=*p++;
	    f->fdata()(k,j,i)=Tensor(tens);
	  }
      fieldH = f;      
      ofield->send(fieldH);
    } else {
      cerr << "Error - 4D nrrd must have 7 entries per channel (1st dim).\n";
    }
    return;
  }

  if (n->dim != 3) {
    cerr << "Can only deal with 3-dimensional scalar fields... sorry.\n";
    return;
  }
  int nx = n->size[0];
  int ny = n->size[1];
  int nz = n->size[2];
  
  Point minP(0,0,0), maxP((nx-1)*n->spacing[0],
			  (ny-1)*n->spacing[1],
			  (nz-1)*n->spacing[2]);	  
  LatVolMesh *lvm = scinew LatVolMesh(nx, ny, nz, minP, maxP);
  LatVolMeshHandle lvmH(lvm);
 
  if (n->type == nrrdTypeChar) {
    LatticeVol<char> *f = 
      scinew LatticeVol<char>(lvm, Field::NODE);
    char *p=(char *)n->data;
    for (k=0; k<nz; k++)
      for (j=0; j<ny; j++)
	for(i=0; i<nx; i++)
	  f->fdata()(k,j,i) = *p++;
    fieldH = f;
  } else if (n->type == nrrdTypeUChar) {
    LatticeVol<unsigned char> *f = 
      scinew LatticeVol<unsigned char>(lvm, Field::NODE);
    unsigned char *p=(unsigned char *)n->data;
    for (k=0; k<nz; k++)
      for (j=0; j<ny; j++)
	for(i=0; i<nx; i++)
	  f->fdata()(k,j,i) = *p++;
    fieldH = f;
  } else if (n->type == nrrdTypeShort) {
    LatticeVol<short> *f = 
      scinew LatticeVol<short>(lvm, Field::NODE);
    short *p=(short *)n->data;
    for (k=0; k<nz; k++)
      for (j=0; j<ny; j++)
	for(i=0; i<nx; i++)
	  f->fdata()(k,j,i) = *p++;
    fieldH = f;
  } else if (n->type == nrrdTypeUShort) {
    LatticeVol<unsigned short> *f = 
      scinew LatticeVol<unsigned short>(lvm, Field::NODE);
    unsigned short *p=(unsigned short *)n->data;
    for (k=0; k<nz; k++)
      for (j=0; j<ny; j++)
	for(i=0; i<nx; i++)
	  f->fdata()(k,j,i) = *p++;
    fieldH = f;
  } else if (n->type == nrrdTypeInt) {
    LatticeVol<int> *f = 
      scinew LatticeVol<int>(lvm, Field::NODE);
    int *p=(int *)n->data;
    for (k=0; k<nz; k++)
      for (j=0; j<ny; j++)
	for(i=0; i<nx; i++)
	  f->fdata()(k,j,i) = *p++;
    fieldH = f;
  } else if (n->type == nrrdTypeUInt) {
    LatticeVol<unsigned int> *f = 
      scinew LatticeVol<unsigned int>(lvm, Field::NODE);
    unsigned int *p=(unsigned int *)n->data;
    for (k=0; k<nz; k++)
      for (j=0; j<ny; j++)
	for(i=0; i<nx; i++)
	  f->fdata()(k,j,i) = *p++;
    fieldH = f;
  } else if (n->type == nrrdTypeFloat) {
    LatticeVol<float> *f = 
      scinew LatticeVol<float>(lvm, Field::NODE);
    float *p=(float *)n->data;
    for (k=0; k<nz; k++)
      for (j=0; j<ny; j++)
	for(i=0; i<nx; i++)
	  f->fdata()(k,j,i) = *p++;
    fieldH = f;
  } else if (n->type == nrrdTypeDouble) {
    LatticeVol<double> *f = 
      scinew LatticeVol<double>(lvm, Field::NODE);
    double *p=(double *)n->data;
    for (k=0; k<nz; k++)
      for (j=0; j<ny; j++)
	for(i=0; i<nx; i++)
	  f->fdata()(k,j,i) = *p++;
    fieldH = f;
  } else {
    cerr << "NrrdToField error - Unrecognized nrrd type.\n";
    return;
  }
  
  ofield->send(fieldH);
}

} // End namespace SCITeem
