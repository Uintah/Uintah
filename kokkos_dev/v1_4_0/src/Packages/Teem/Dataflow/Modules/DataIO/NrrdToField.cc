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
#include <Core/Datatypes/MaskedLatVolField.h>
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


NrrdToField::NrrdToField(const string& id):Module("NrrdToField", id, Filter, "DataIO", "Teem")
{
}

NrrdToField::~NrrdToField()
{
}

#define COPY_INTO_FIELD_FROM_NRRD(type, dataat) \
    LatVolField<type> *f = \
      scinew LatVolField<type>(lvm, dataat); \
    type *p=(type *)n->data; \
    for (k=0; k<nz; k++) \
      for (j=0; j<ny; j++) \
	for(i=0; i<nx; i++) \
	  f->fdata()(k,j,i) = *p++; \
    fieldH = f

void NrrdToField::execute()
{
  NrrdDataHandle ninH;
  inrrd = (NrrdIPort *)get_iport("Nrrd");
  ofield = (FieldOPort *)get_oport("Field");

  if (!inrrd) {
    postMessage("Unable to initialize "+name+"'s iport\n");
    return;
  }
  if (!ofield) {
    postMessage("Unable to initialize "+name+"'s oport\n");
    return;
  }

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
    if (n->axis[0].size == 7) {
      if (n->type != nrrdTypeFloat) {
	cerr << "Error - NrrdToField can only convert float data for tensors.\n";
	return;
      }
      // the Nrrd assumed samples at nodes and gave min/max accordingly
      // but we want to think of those samples as centers of cells, so
      // we need a different mesh
      Point minP(0,0,0);
      Point maxP((n->axis[1].size+1)*n->axis[1].spacing, 
		 (n->axis[2].size+1)*n->axis[2].spacing,
		 (n->axis[3].size+1)*n->axis[3].spacing);
      int nx = n->axis[1].size;
      int ny = n->axis[2].size;
      int nz = n->axis[3].size;
      LatVolMesh *lvm = scinew LatVolMesh(nx+1, ny+1, nz+1, minP, maxP);
      MaskedLatVolField<Tensor> *f =
	scinew MaskedLatVolField<Tensor>(lvm, Field::CELL);
      float *p=(float *)n->data;
      Array1<double> tens(6);
      for (k=0; k<nz; k++)
	for (j=0; j<ny; j++)
	  for (i=0; i<nx; i++) {
	    if (*p++ > .5) f->mask()(k,j,i)=1;
	    else f->mask()(k,j,i)=0;
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
    cerr << "NrrdToField error: nrrd->dim="<<n->dim<<"\n";
    cerr << "  Can only deal with 3-dimensional scalar fields... sorry.\n";
    return;
  }
  int nx = n->axis[0].size;
  int ny = n->axis[1].size;
  int nz = n->axis[2].size;
  
  for (i=0; i<3; i++)
    if (!(AIR_EXISTS(n->axis[i].min) && AIR_EXISTS(n->axis[i].max)))
      nrrdAxisSetMinMax(n, i);

  Point minP(n->axis[0].min, n->axis[1].min, n->axis[2].min);
  Point maxP(n->axis[0].max, n->axis[1].max, n->axis[2].max);

  LatVolMesh *lvm = scinew LatVolMesh(nx, ny, nz, minP, maxP);
  LatVolMeshHandle lvmH(lvm);
  SCIRun::Field::data_location dataat = Field::NODE;
  if (n->axis[0].center == nrrdCenterCell) dataat = Field::CELL;

  if (n->type == nrrdTypeChar) {
    COPY_INTO_FIELD_FROM_NRRD(char, dataat);
  } else if (n->type == nrrdTypeUChar) {
    COPY_INTO_FIELD_FROM_NRRD(unsigned char, dataat);
  } else if (n->type == nrrdTypeShort) {
    COPY_INTO_FIELD_FROM_NRRD(short, dataat);
  } else if (n->type == nrrdTypeUShort) {
    COPY_INTO_FIELD_FROM_NRRD(unsigned short, dataat);
  } else if (n->type == nrrdTypeInt) {
    COPY_INTO_FIELD_FROM_NRRD(int, dataat);
  } else if (n->type == nrrdTypeUInt) {
    COPY_INTO_FIELD_FROM_NRRD(unsigned int, dataat);
  } else if (n->type == nrrdTypeFloat) {
    COPY_INTO_FIELD_FROM_NRRD(float, dataat);
  } else if (n->type == nrrdTypeDouble) {
    COPY_INTO_FIELD_FROM_NRRD(double, dataat);
  } else {
    cerr << "NrrdToField error - Unrecognized nrrd type.\n";
    return;
  }
  
  ofield->send(fieldH);
}

} // End namespace SCITeem
