/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/



/*
 *  MaskLatVolWithTriSurf.cc:  MaskLatVolWithTriSurf two point clouds
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   March 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Containers/Array1.h>
#include <Core/Geometry/BBox.h>
#include <Core/Datatypes/LatVolField.h>
#include <Core/Datatypes/TriSurfField.h>
#include <iostream>

using std::cerr;

namespace SCIRun {

class MaskLatVolWithTriSurf : public Module
{
private:
public:
  MaskLatVolWithTriSurf(GuiContext* ctx);
  virtual ~MaskLatVolWithTriSurf();
  virtual void execute();
};


DECLARE_MAKER(MaskLatVolWithTriSurf)

MaskLatVolWithTriSurf::MaskLatVolWithTriSurf(GuiContext* ctx)
  : Module("MaskLatVolWithTriSurf", ctx, Filter, "FieldsData", "SCIRun")
{
}

MaskLatVolWithTriSurf::~MaskLatVolWithTriSurf()
{
}

void
MaskLatVolWithTriSurf::execute()
{
  FieldIPort *latvol = (FieldIPort *) get_iport("LatVolField");
  FieldIPort *trisurf = (FieldIPort *) get_iport("TriSurfField");

  FieldHandle latvolH, trisurfH;
  LatVolMeshHandle latvolM;
  TriSurfMeshHandle trisurfM;
  
  if (!latvol->get(latvolH)) {
    warning("No input on LatVol port.");
    return;
  }
  latvolM = dynamic_cast<LatVolMesh*>(latvolH->mesh().get_rep());
  if (!latvolH.get_rep()) {
    error("Input field was not a LatVol.");
    return;
  }

  if (!trisurf->get(trisurfH)) {
    warning("No input on TriSurf port.");
    return;
  }
  trisurfM = dynamic_cast<TriSurfMesh*>(trisurfH->mesh().get_rep());
  if (!trisurfM.get_rep()) {
    error("Input field was not a TriSurf.");
    return;
  }

  FieldOPort *omask = (FieldOPort *) get_oport("LatVol Mask");
  LatVolField<char> *mask=scinew LatVolField<char>(latvolM, 1);

  TriSurfMesh::Face::iterator fiter; 
  TriSurfMesh::Face::iterator fiter_end; 
  TriSurfMesh::Face::size_type nfaces;
  TriSurfMesh::Node::array_type fac_nodes(3);
  trisurfM->begin(fiter);
  trisurfM->end(fiter_end);
  trisurfM->size(nfaces);
  Array1<BBox> faceBBox(nfaces);
  int fidx=0;
  int i;
  BBox surfBBox;
  while(fiter != fiter_end) {
    trisurfM->get_nodes(fac_nodes, *fiter);
    for (i=0; i<3; i++) {
      Point p;
      trisurfM->get_center(p, fac_nodes[i]);
      faceBBox[fidx].extend(p);
      surfBBox.extend(p);
    }
    ++fidx;
    ++fiter;
  }
  LatVolMesh::Node::iterator niter;
  LatVolMesh::Node::iterator niter_end;
  latvolM->begin(niter);
  latvolM->end(niter_end);
  while (niter != niter_end) {
    Point p;
    latvolM->get_center(p, *niter);
    if (!surfBBox.inside(p)) { mask->set_value(0, *niter); ++niter; continue; }
    int ncrossings=0;
    trisurfM->begin(fiter);
    while (fiter != fiter_end) {
      if (faceBBox[*fiter].min().x() < p.x() &&
	  faceBBox[*fiter].max().x() > p.x() &&
	  faceBBox[*fiter].min().y() < p.y() &&
	  faceBBox[*fiter].max().y() > p.y()) {
	trisurfM->get_nodes(fac_nodes, *fiter);
	Point p1, p2, p3;
	trisurfM->get_center(p1, fac_nodes[0]);
	trisurfM->get_center(p2, fac_nodes[1]);
	trisurfM->get_center(p3, fac_nodes[2]);

	Vector e1(p2-p1);
	Vector e2(p3-p1);
	Vector pvec(Cross(Vector(0,0,1), e2));
	double det=Dot(e1, pvec);

	if(det>1.e-9 || det<-1.e-9) {
	  double idet=1.0/det;
	  Vector tvec(p-p1);
	  double u=Dot(tvec, pvec)*idet;
	  if (u>=0.0 && u<=1.0) {
	    Vector qvec(Cross(tvec, e1));
	    double v=Dot(Vector(0,0,1), qvec)*idet;
	    if (v>=0.0 && u+v<=1.0) {
	      double t=Dot(e2, qvec)*idet;
	      if (t>0) {
		ncrossings++;
	      }
	    }
	  }
	}
      }
      ++fiter;
    }
    if (ncrossings % 2) mask->set_value(1, *niter);
    else mask->set_value(0, *niter);
    ++niter;
  }

  // go through all of the nodes in the LatVolField and see if they're
  // inside the TriSurf (count face crossings)

  omask->send(FieldHandle(mask));
}
} // End namespace SCIRun
