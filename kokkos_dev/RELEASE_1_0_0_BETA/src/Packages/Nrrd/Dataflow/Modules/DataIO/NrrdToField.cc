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
#include <Nrrd/Dataflow/Ports/NrrdPort.h>
#include <Dataflow/Ports/FieldPort.h>
//#include <Core/Datatypes/LatticeVol.h>
#include <Core/Malloc/Allocator.h>
#include <iostream>

using std::cerr;
using std::endl;

namespace SCINrrd {

using namespace SCIRun;

class NrrdToField : public Module {
  NrrdIPort* inrrd;
  FieldOPort* ofield;
public:
  NrrdToField(const clString& id);
  virtual ~NrrdToField();
  virtual void execute();
};

extern "C" Module* make_NrrdToField(const clString& id) {
  return new NrrdToField(id);
}


NrrdToField::NrrdToField(const clString& id):Module("NrrdToField", id, Filter)
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

  if (ninH->nrrd->dim != 3) {
    cerr << "Can only deal with 3-dimensional scalar fields... sorry.\n";
    return;
  }

  int nx = ninH->nrrd->size[0];
  int ny = ninH->nrrd->size[1];
  int nz = ninH->nrrd->size[2];
  
  Point minP, maxP;
  FieldHandle sfH;

  minP.x(ninH->nrrd->axisMin[0]);
  if ((minP.x()<1 || minP.x()>-1)) {	// !NaN
    minP.y(ninH->nrrd->axisMin[1]);
    minP.z(ninH->nrrd->axisMin[2]);
    maxP.x(ninH->nrrd->axisMax[0]);
    maxP.y(ninH->nrrd->axisMax[1]);
    maxP.z(ninH->nrrd->axisMax[2]);
  } else {
    minP=Point(0,0,0);
    maxP=Point(nx-1, ny-1, nz-1);
  }
 
// waiting for Fields... 
#if 0
  int i, j, k;
  if (ninH->nrrd->type == nrrdTypeChar) {
    ScalarFieldRGchar *ifc = new ScalarFieldRGchar(nx,ny,nz);
    ifc->set_bounds(minP, maxP);
    char *p=(char *)ninH->nrrd->data;
    for (k=0; k<nz; k++) {
      for (j=0; j<ny; j++) {
	for(i=0; i<nx; i++) {
	  ifc->grid(i,j,k) = *p++;
	}
      }
    }
    sfH = ifc;
  } else if (ninH->nrrd->type == nrrdTypeUChar) {
    ScalarFieldRGuchar *ifu = new ScalarFieldRGuchar(nx,ny,nz);
    ifu->set_bounds(minP, maxP);
    uchar *p=(uchar *)ninH->nrrd->data;
    for (k=0; k<nz; k++) {
      for (j=0; j<ny; j++) {
	for(i=0; i<nx; i++) {
	  ifu->grid(i,j,k) = *p++;
	}
      }
    }
    sfH = ifu;
  } else if (ninH->nrrd->type == nrrdTypeShort) {
    ScalarFieldRGshort *ifs = new ScalarFieldRGshort(nx,ny,nz);
    ifs->set_bounds(minP, maxP);
    short int *p=(short int *)ninH->nrrd->data;
    for (k=0; k<nz; k++) {
      for (j=0; j<ny; j++) {
	for(i=0; i<nx; i++) {
	  ifs->grid(i,j,k) = *p++;
	}
      }
    }
    sfH = ifs;
  } else if (ninH->nrrd->type == nrrdTypeUShort) {
    ScalarFieldRGushort *ifus = new ScalarFieldRGushort(nx,ny,nz);
    ifus->set_bounds(minP, maxP);
    short int *p=(short int *)ninH->nrrd->data;
    for (k=0; k<nz; k++) {
      for (j=0; j<ny; j++) {
	for(i=0; i<nx; i++) {
	  ifus->grid(i,j,k) = *p++;
	}
      }
    }
    sfH = ifus;
  } else if (ninH->nrrd->type == nrrdTypeInt) {
    ScalarFieldRGint *ifi = new ScalarFieldRGint(nx,ny,nz);
    ifi->set_bounds(minP, maxP);
    int *p=(int *)ninH->nrrd->data;
    for (k=0; k<nz; k++) {
      for (j=0; j<ny; j++) {
	for(i=0; i<nx; i++) {
	  ifi->grid(i,j,k) = *p++;
	}
      }
    }
    sfH = ifi;
  } else if (ninH->nrrd->type == nrrdTypeFloat) {
    ScalarFieldRGfloat *iff = new ScalarFieldRGfloat(nx,ny,nz);
    iff->set_bounds(minP, maxP);
    float *p=(float *)ninH->nrrd->data;
    for (k=0; k<nz; k++) {
      for (j=0; j<ny; j++) {
	for(i=0; i<nx; i++) {
	  iff->grid(i,j,k) = *p++;
	}
      }
    }
    sfH = iff;
  } else if (ninH->nrrd->type == nrrdTypeDouble) {
    ScalarFieldRGdouble *ifd = new ScalarFieldRGdouble(nx,ny,nz);
    ifd->set_bounds(minP, maxP);
    double *p=(double *)ninH->nrrd->data;
    for (k=0; k<nz; k++) {
      for (j=0; j<ny; j++) {
	for(i=0; i<nx; i++) {
	  ifd->grid(i,j,k) = *p++;
	}
      }
    }
    sfH = ifd;
  } else {
    cerr << "NrrdToField error - Unrecognized nrrd type.\n";
    return;
  }
#endif
  
  ofield->send(sfH);
}

} // End namespace SCINrrd
