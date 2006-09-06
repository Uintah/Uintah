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
 *  IntegrateCurrent: Compute current through a surface
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   January 2002
 *
 *  Copyright (C) 2002 SCI Group
 */

#include <Core/Basis/TriLinearLgn.h>
#include <Core/Datatypes/TriSurfMesh.h>
#include <Core/Basis/TetLinearLgn.h>
#include <Core/Datatypes/TetVolMesh.h>
#include <Core/Datatypes/GenericField.h>
#include <Dataflow/Network/Ports/FieldPort.h>
#include <Dataflow/GuiInterface/GuiVar.h>
#include <iostream>
#include <sstream>

namespace BioPSE {

using namespace SCIRun;

class IntegrateCurrent : public Module {
  typedef TetLinearLgn<Vector>                                  TFDVectorBasis;
  typedef TetLinearLgn<int>                                     TFDintBasis;
  typedef TetVolMesh<TetLinearLgn<Point> >                      TVMesh;
  typedef GenericField<TVMesh, TFDVectorBasis, vector<Vector> > TVFieldV;
  typedef GenericField<TVMesh, TFDintBasis,    vector<int> >    TVFieldI;
  typedef TriSurfMesh<TriLinearLgn<Point> >                     TSMesh;

  GuiDouble current_;
public:
  IntegrateCurrent(GuiContext *context);
  virtual ~IntegrateCurrent();
  virtual void execute();
};

DECLARE_MAKER(IntegrateCurrent)


IntegrateCurrent::IntegrateCurrent(GuiContext *context)
  : Module("IntegrateCurrent", context, Filter, "Forward", "BioPSE"),
    current_(context->subVar("current"))
{
}

IntegrateCurrent::~IntegrateCurrent()
{
}

void
IntegrateCurrent::execute()
{
  FieldHandle efieldH, sigmasH, trisurfH;
  if (!get_input_handle("TetMesh EField", efieldH)) return;
  if (!get_input_handle("TetMesh Sigmas", sigmasH)) return;
  if (!get_input_handle("TriSurf", trisurfH)) return;

  if (efieldH->mesh().get_rep() != sigmasH->mesh().get_rep()) {
    error("EField and Sigma Field need to have the same mesh.");
    return;
  }

  TVFieldV *efield = dynamic_cast<TVFieldV*>(efieldH.get_rep());
  if (!efield) {
    error("EField isn't a TetVolField<Vector>.");
    return;
  }
  TVFieldI *sigmas = dynamic_cast<TVFieldI*>(sigmasH.get_rep());
  if (!sigmas) {
    error("Sigmas isn't a TetVolField<int>.");
    return;
  }
  TSMesh *tris = dynamic_cast<TSMesh*>(trisurfH->mesh().get_rep());
  if (!tris) {
    error("Not a TriSurf.");
    return;
  }

  vector<pair<string, Tensor> > conds;
  if (!sigmasH->get_property("conductivity_table", conds)) {
    error("No conductivity_table found in Sigmas.");
    return;
  }

  // for each face in tris, find its area, centroid, and normal
  // for that centroid, look up its sigma and efield in the tetvol fields
  // compute (sigma * efield * area) and dot it with the face normal
  // sum those up for all tris

  TSMesh::Face::iterator fi, fe;
  tris->begin(fi);
  tris->end(fe);
  double current=0;
  TSMesh::Node::array_type nodes;
  double total_area=0;
  while (fi != fe) {
    Point center;
    tris->get_center(center, *fi);
    double area = tris->get_area(*fi);
    total_area += area;
    tris->get_nodes(nodes, *fi);
    Point p0, p1, p2;
    tris->get_center(p0, nodes[0]);
    tris->get_center(p1, nodes[1]);
    tris->get_center(p2, nodes[2]);
    Vector normal(Cross(p2-p1,p2-p0));
    normal.normalize();
    TVMesh::Cell::index_type tet;
    if (!efield->get_typed_mesh()->locate(tet, center)) {
      error("Trisurf centroid was not located in tetvolmesh.");
      return;
    }
    Vector e = efield->value(tet);
    int sigma_idx = sigmas->value(tet);
    Tensor s(conds[sigma_idx].second);
    
    // compute sigma * e
    Vector c(s.mat_[0][0]*e.x()+s.mat_[0][1]*e.y()+s.mat_[0][2]*e.z(),
	     s.mat_[1][0]*e.x()+s.mat_[1][1]*e.y()+s.mat_[1][2]*e.z(),
	     s.mat_[2][0]*e.x()+s.mat_[2][1]*e.y()+s.mat_[2][2]*e.z());
    current += fabs(Dot(c,normal)) * area;
    ++fi;
  }
  current_.set(current);
}

} // End namespace BioPSE
