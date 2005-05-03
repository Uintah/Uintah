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
 *  PointLatticeMap.cc:  
 *   Builds mapping matrix that projects data from a PointCloud to a LatVol
 *
 *  Written by:
 *   McKay Davis
 *   Scientific Computing and Imaging INstitute
 *   University of Utah
 *   May 2005
 *
 *  Copyright (C) 2005 SCI Group
 */

#include <Core/GuiInterface/UIvar.h>
#include <Core/Util/DynamicCompilation.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Core/Thread/CrowdMonitor.h>
#include <Dataflow/Widgets/BoxWidget.h>
#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/FieldInterface.h>
#include <Core/Datatypes/Clipper.h>
#include <Core/Datatypes/SparseRowMatrix.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/PointCloudField.h>
#include <Core/Datatypes/LatVolField.h>

#include <Core/Containers/StringUtil.h>
#include <iostream>
#include <stack>

namespace SCIRun {

using std::stack;

class PointLatticeMap : public Module
{
private:
  FieldIPort *		iport1_;
  FieldIPort *		iport2_;
  MatrixOPort *		oport_;
  UIint			ui_num_neighbors_;
  int			pcf_generation_;
  int			lvf_generation_;
  int			num_neighbors_;
public:
  PointLatticeMap(GuiContext* ctx);
  virtual ~PointLatticeMap();

  virtual void execute();
};


DECLARE_MAKER(PointLatticeMap)

PointLatticeMap::PointLatticeMap(GuiContext* ctx)
  : Module("PointLatticeMap", ctx, Filter, "FieldsData", "SCIRun"),
    ui_num_neighbors_(ctx->subVar("num_neighbors")),
    pcf_generation_(-1),
    lvf_generation_(-1),
    num_neighbors_(0)
{
}


PointLatticeMap::~PointLatticeMap()
{
}


void
PointLatticeMap::execute()
{
  iport1_ = (FieldIPort *)get_iport("PointCloudField");
  iport2_ = (FieldIPort *)get_iport("LatVolField");
  oport_ = (MatrixOPort *)get_oport("MappingMatrix");


  FieldHandle pcf;
  iport1_->get(pcf);
  if (!pcf.get_rep()) {
    error("No input field to port 1.");
    return;
  }

  FieldHandle lvf;
  iport2_->get(lvf);
  if (!lvf.get_rep()) {
    error("No input field to port 2.");
    return;
  }

    
  if (pcf->get_type_description()->get_name().find("PointCloudField")) {
    error("Field connected to port 1 must be PointCloudField.");
    return;
  }

  if (lvf->get_type_description()->get_name().find("LatVolField")) {
    error("Field connected to port 2 must be LatVolField.");
    return;
  }

  if (pcf->generation == pcf_generation_ &&
      lvf->generation == lvf_generation_ &&
      ui_num_neighbors_() == num_neighbors_) return;

  pcf_generation_ = pcf->generation;
  lvf_generation_ = lvf->generation;

  PointCloudMeshHandle pcm = (PointCloudMesh *)(pcf->mesh().get_rep());
  LatVolMeshHandle lvm = (LatVolMesh *)(lvf->mesh().get_rep());

  vector<unsigned int> lvm_dim, pcm_dim;
  lvm->get_dim(lvm_dim);
  pcm->get_dim(pcm_dim);

  typedef map<double, unsigned int> point_distances_t;
  vector<point_distances_t> mapping;
  
  LatVolMesh::Node::size_type lvmns;
  lvm->size(lvmns);
  mapping.resize(lvmns);
  
  LatVolMesh::Node::iterator lvmn, lvmne;
  lvm->begin(lvmn);
  lvm->end(lvmne);

  PointCloudMesh::Node::iterator pcmn, pcmne;
  pcm->end(pcmne);

  PointCloudMesh::Node::size_type pcmns;
  pcm->size(pcmns);

  Point pcp, lvp;
  double d;

  num_neighbors_ = Clamp(ui_num_neighbors_(), 1, pcmns);

  while (lvmn != lvmne) {
    point_distances_t &point_mapping = mapping[unsigned(*lvmn)];
    pcm->begin(pcmn);
    while (pcmn != pcmne) {
      lvm->get_point(lvp, *lvmn);
      pcm->get_point(pcp, *pcmn);
      d = (pcp-lvp).length2();
      if (int(point_mapping.size())  < num_neighbors_) {
	point_mapping.insert(make_pair(d, (*pcmn).index_));
      } else if ((*point_mapping.rbegin()).first > d) {
	point_distances_t::iterator last = point_mapping.end();
	last--;
	point_mapping.erase(last);
	point_mapping.insert(make_pair(d, (*pcmn).index_));
      }
      ++pcmn;
    }
    ++lvmn;
  }
  
  
  int *rows = scinew int[mapping.size()+1];
  int *cols = scinew int[mapping.size()*num_neighbors_];
  double *data = scinew double[mapping.size()*num_neighbors_];
  unsigned int rowcount, r, i = 0;
  double total;
  
  typedef map<unsigned int, double> point_neighbors_t;
  for (r = 0; r < mapping.size(); r++) {
    rows[r] = i;
    point_distances_t::iterator pb = mapping[r].begin(), pe = mapping[r].end();
    point_neighbors_t neighbors;
    total = 0.0;
    rowcount = 0;
    while (pb != pe) {
      neighbors[pb->second] = pb->first;
      total += pb->first;
      ++pb;
    }
    if (total == 0.0) 
      total = 1.0;
    point_neighbors_t::iterator nb = neighbors.begin(), ne = neighbors.end();
    while (nb != ne) {
      cols[i] = nb->first;
      data[i] = 1.0 - nb->second/total;
      ++i;
      ++nb;
    }
  }  rows[r] = i;
  SparseRowMatrix *matrix = 
    scinew SparseRowMatrix(mapping.size(), pcmns, rows, cols, i, data);

//  matrix->validate();
//   DenseMatrix *matrix = scinew DenseMatrix(mapping.size(), idx);
//   matrix->zero();
//   for (unsigned int n = 0; n < mapping.size(); n++) {
//     point_distances_t::iterator pb = mapping[n].begin(), pe = mapping[n].end();
//     while (pb != pe) {
//       matrix->put(n, (*pb).second, (*pb).first);
//       pb++;
//     }
//   }
//  matrix->dense()->print();

  oport_->send(matrix);


}


}
