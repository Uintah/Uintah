/*
 *  FloodFillNewValue.cc:
 *
 *   Written by:
 *   David Weinstein
 *   May 2002
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Datatypes/HexVolField.h>
#include <Core/Datatypes/PointCloudField.h>
#include <Core/Geometry/Vector.h>

#include <Packages/CardioWave/share/share.h>

#include <deque>

namespace CardioWave {

using namespace SCIRun;
using std::deque;

class CardioWaveSHARE FloodFillNewValue : public Module {
public:
  FloodFillNewValue(GuiContext *context);
  virtual ~FloodFillNewValue();
  virtual void execute();
};


DECLARE_MAKER(FloodFillNewValue)


FloodFillNewValue::FloodFillNewValue(GuiContext *context)
  : Module("FloodFillNewValue", context, Source, 
	   "CreateModel", "CardioWave")
{
}

FloodFillNewValue::~FloodFillNewValue(){
}

void FloodFillNewValue::execute(){
  string pname;

  pname="Volume";
  FieldIPort *ivol = (FieldIPort*)get_iport(pname);
  if (!ivol) {
    error("Unable to initialize iport '" + pname + "'.");
    return;
  }
  FieldHandle volH;
  if (!ivol->get(volH) || !volH.get_rep()) {
    error("Not a valid field in '" + pname + "'.");
    return;
  }
  HexVolField<int> *hvf =
    dynamic_cast<HexVolField<int> *>(volH.get_rep());
  if (!hvf) {
    error("Input was not a valid HexVolField<int>.");
    return;
  }

  pname="Point";
  FieldIPort *ipt = (FieldIPort*)get_iport(pname);
  if (!ipt) {
    error("Unable to initialize iport '"+ pname + "'.");
    return;
  }
  FieldHandle ptH;
  if (!ipt->get(ptH) || !ptH.get_rep()) {
    error("Not a valid field in '" + pname + "'.");
    return;
  }
  PointCloudField<double> *pcf =
    dynamic_cast<PointCloudField<double> *>(ptH.get_rep());
  if (!pcf) {
    error("Input was not a PointCloudField<double>.");
    return;
  }
  if (pcf->fdata().size() == 0) {
    error("Input point cloud was empty.");
  }

  pname="FloodFilled Volume";
  FieldOPort *ovol = (FieldOPort*)get_oport(pname);
  if (!ovol) {
    error("Unable to initialize output port '"+pname+"'.");
    return;
  }

  PointCloudMesh::Node::index_type node0(0);
  Point p;
  pcf->get_typed_mesh()->get_center(p, node0);
  int data_at = hvf->data_at();
  
  if (data_at == Field::NODE) {
    hvf->get_typed_mesh()->synchronize(Mesh::LOCATE_E);
    HexVolMesh::Node::index_type loc;
    if (!hvf->get_typed_mesh()->locate(loc, p)) {
      ovol->send(volH);
      return;
    }
    int new_val = pcf->fdata()[0];
    int old_val;
    if (!hvf->value(old_val, loc)) {
      error("There's no datavalue in the HexVol at this Node location, even though the call to locate() succeeded.");
      return;
    }
    
    // the point was found in HexVolMesh node "loc" -- need to flood fill all
    // nodes connected to "loc" that have value "old_val" -- change them to 
    // have the  with the new value "new_val".
    //
    // implement this by copying the HexVolField into a new field -- nodes with
    // data value "old_val" will be set to 1, all others will be set to 0.
    // the origian node (loc) will then be set to 2, and we'll start our flood
    // fill by pushing "locl" onto a stack.
    // while the stack isn't empty, pop the top of the stack into "curr"
    // look at all of the neighbors of "curr".  For any neighbor that has a
    // value "1", change its value to "2", set the corresponding location in
    // the original volume to be "new_val", and push the neighbor's location
    // onto the stack.
    // when the stack is empty, we're done.
    
    volH.detach();
    hvf = dynamic_cast<HexVolField<int> *>(volH.get_rep());
    
    HexVolField<char> *mask_vol = 
      scinew HexVolField<char>(hvf->get_typed_mesh(), Field::NODE);
    
  // set up the mask volume
    HexVolMesh::Node::iterator curr_node, last_node;
    hvf->get_typed_mesh()->begin(curr_node);
    hvf->get_typed_mesh()->end(last_node);
    while(curr_node != last_node) {
      int curr_val;
      if (hvf->value(curr_val, *curr_node) && curr_val == old_val)
	mask_vol->set_value(1, *curr_node);
    else
      mask_vol->set_value(0, *curr_node);
      ++curr_node;
    }
    
    // make our queue
    deque<HexVolMesh::Node::index_type> Q;
    hvf->set_value(new_val, loc);
    mask_vol->set_value(2, loc);
    Q.push_back(loc);


    hvf->get_typed_mesh()->synchronize(Mesh::NODE_NEIGHBORS_E);
    
    // flood fill
    while (!Q.empty()) {
      HexVolMesh::Node::index_type curr = Q.front();
      Q.pop_front();
      HexVolMesh::Node::array_type nbrs;
      hvf->get_typed_mesh()->get_neighbors(nbrs, curr);
      HexVolMesh::Node::array_type::iterator iter = nbrs.begin();
      while (iter != nbrs.end()) {
	char nbr_val;
	if (mask_vol->value(nbr_val, *iter) && nbr_val == 1) {
	  hvf->set_value(new_val, *iter);
	  mask_vol->set_value(2, *iter);
	  Q.push_back(*iter);
      }
	++iter;
      }
    }
    
    delete mask_vol;
  } else {
    HexVolMesh::Cell::index_type loc;
    if (!hvf->get_typed_mesh()->locate(loc, p)) {
      ovol->send(volH);
      return;
    }
    int new_val = pcf->fdata()[0];
    int old_val;
    if (!hvf->value(old_val, loc)) {
      error("There's no datavalue in the HexVol at this Cell location, even though the call to locate() succeeded.");
      return;
    }
    
    // the point was found in HexVolMesh cell "loc" -- need to flood fill all
    // cells connected to "loc" that have value "old_val" -- change them to 
    // have the  with the new value "new_val".
    //
    // implement this by copying the HexVolField into a new field -- cells with
    // data value "old_val" will be set to 1, all others will be set to 0.
    // the origian cell (loc) will then be set to 2, and we'll start our flood
    // fill by pushing "locl" onto a stack.
    // while the stack isn't empty, pop the top of the stack into "curr"
    // look at all of the neighbors of "curr".  For any neighbor that has a
    // value "1", change its value to "2", set the corresponding location in
    // the original volume to be "new_val", and push the neighbor's location
    // onto the stack.
    // when the stack is empty, we're done.
    
    volH.detach();
    hvf = dynamic_cast<HexVolField<int> *>(volH.get_rep());
    
    HexVolField<char> *mask_vol = 
      scinew HexVolField<char>(hvf->get_typed_mesh(), Field::CELL);
    
  // set up the mask volume
    HexVolMesh::Cell::iterator curr_cell, last_cell;
    hvf->get_typed_mesh()->begin(curr_cell);
    hvf->get_typed_mesh()->end(last_cell);
    while(curr_cell != last_cell) {
      int curr_val;
      if (hvf->value(curr_val, *curr_cell) && curr_val == old_val)
	mask_vol->set_value(1, *curr_cell);
    else
      mask_vol->set_value(0, *curr_cell);
      ++curr_cell;
    }
    
    // make our queue
    deque<HexVolMesh::Cell::index_type> Q;
    hvf->set_value(new_val, loc);
    mask_vol->set_value(2, loc);
    Q.push_back(loc);

    // flood fill
    while (!Q.empty()) {
      HexVolMesh::Cell::index_type curr = Q.front();
      Q.pop_front();
      HexVolMesh::Cell::array_type nbrs;
      hvf->get_typed_mesh()->get_neighbors(nbrs, curr);
      HexVolMesh::Cell::array_type::iterator iter = nbrs.begin();
      while (iter != nbrs.end()) {
	char nbr_val;
	if (mask_vol->value(nbr_val, *iter) && nbr_val == 1) {
	  hvf->set_value(new_val, *iter);
	  mask_vol->set_value(2, *iter);
	  Q.push_back(*iter);
      }
	++iter;
      }
    }
    
    delete mask_vol;
  }
  ovol->send(volH);
  }
} // End namespace CardioWave
