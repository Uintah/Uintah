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
 *  TransformField.cc:  Rotate and flip field to get it into "standard" view
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   March 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Core/Containers/String.h>
#include <Dataflow/Network/Module.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Geometry/Transform.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/MiscMath.h>
#include <iostream>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

using std::cerr;

namespace SCIRun {


class TransformField : public Module
{

  void MatToTransform(MatrixHandle mH, Transform& t);

public:
  TransformField(const clString& id);
  virtual ~TransformField();

  virtual void execute();
};


extern "C" Module* make_TransformField(const clString& id) {
  return new TransformField(id);
}


TransformField::TransformField(const clString& id)
  : Module("TransformField", id, Source, "Fields", "SCIRun")
{
}



TransformField::~TransformField()
{
}



void
TransformField::MatToTransform(MatrixHandle mH, Transform& t)
{
  double a[16];
  double *p=&(a[0]);
  for (int i=0; i<4; i++)
    for (int j=0; j<4; j++)
      *p++=(*mH.get_rep())[i][j];
  t.set(a);
}


void
TransformField::execute()
{
#if 0
  FieldHandle sfIH;
  iport->get(sfIH);
  if (!sfIH.get_rep()) return;
  FieldRGBase *sfrgb;
  if ((sfrgb=sfIH->getRGBase()) == 0) return;

  MatrixHandle mIH;
  imat->get(mIH);
  if (!mIH.get_rep()) return;
  if ((mIH->nrows() != 4) || (mIH->ncols() != 4)) return;
  Transform t;
  MatToTransform(mIH, t);

  FieldRGdouble *ifd, *ofd;
  FieldRGfloat *iff, *off;
  FieldRGint *ifi, *ofi;
  FieldRGshort *ifs, *ofs;
  FieldRGuchar *ifu, *ofu;
  FieldRGchar *ifc, *ofc;
    
  FieldRGBase *ofb;

  ifd=sfrgb->getRGDouble();
  iff=sfrgb->getRGFloat();
  ifi=sfrgb->getRGInt();
  ifs=sfrgb->getRGShort();
  ifu=sfrgb->getRGUchar();
  ifc=sfrgb->getRGChar();
    
  ofd=0;
  off=0;
  ofs=0;
  ofi=0;
  ofc=0;

  int nx=sfrgb->nx;
  int ny=sfrgb->ny;
  int nz=sfrgb->nz;
  Point min;
  Point max;
  sfrgb->get_bounds(min, max);
  if (ifd) {
    ofd=scinew FieldRGdouble(nx, ny, nz); 
    ofb=ofd;
  } else if (iff) {
    off=scinew FieldRGfloat(nx, ny, nz); 
    ofb=off;
  } else if (ifi) {
    ofi=scinew FieldRGint(nx, ny, nz); 
    ofb=ofi;
  } else if (ifs) {
    ofs=scinew FieldRGshort(nx, ny, nz); 
    ofb=ofs;
  } else if (ifu) {
    ofu=scinew FieldRGuchar(nx, ny, nz); 
    ofb=ofu;
  } else if (ifc) {
    ofc=scinew FieldRGchar(nx, ny, nz); 
    ofb=ofc;
  }
  ofb->set_bounds(Point(min.x(), min.y(), min.z()), 
		  Point(max.x(), max.y(), max.z()));
  for (int i=0; i<nx; i++)
    for (int j=0; j<ny; j++)
      for (int k=0; k<nz; k++) {
	Point oldp(sfrgb->get_point(i,j,k));
	Point newp(t.unproject(oldp));
	double val=0;
	if (ifd) { 
	  ifd->interpolate(newp, val); 
	  ofd->grid(i,j,k)=val;
	} else if (iff) {
	  iff->interpolate(newp, val);
	  off->grid(i,j,k)=(float)val;
	} else if (ifi) {
	  ifi->interpolate(newp, val);
	  ofi->grid(i,j,k)=(int)val;
	} else if (ifs) {
	  ifs->interpolate(newp, val);
	  ofs->grid(i,j,k)=(short)val;
	} else if (ifu) {
	  ifu->interpolate(newp, val);
	  ofu->grid(i,j,k)=(unsigned char)val;
	} else if (ifi) {
	  ifc->interpolate(newp, val);
	  ofc->grid(i,j,k)=(char)val;
	}
      }
  FieldHandle sfOH(ofb);
  oport->send(sfOH);
#endif
}

} // End namespace SCIRun

