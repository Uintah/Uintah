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
 *  EditField.cc:
 *
 *  Written by:
 *   moulding
 *   April 22, 2001
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

#include <Dataflow/share/share.h>

#include <Core/Datatypes/MeshBase.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Core/Datatypes/GenericField.h>
#include <Core/Datatypes/LatticeVol.h>
#include <Core/Datatypes/TetVol.h>
#include <Core/Datatypes/TriSurf.h>
#include <Core/Datatypes/PointCloud.h>
#include <Core/Geometry/Transform.h>
#include <Core/Thread/CrowdMonitor.h>
#include <Dataflow/Modules/Fields/EditField.h>
#include <Dataflow/Widgets/ScaledBoxWidget.h>
#include <Dataflow/Network/NetworkEditor.h>
#include <map>

namespace SCIRun {

using std::pair;

class PSECORESHARE EditField : public Module {
public:
  // out is the output field
  // in is the input field

  GuiString numnodes_;   // the in number of nodes 

  GuiString fldname_;    // the out property "name"
  GuiString typename_;   // the out field type
  GuiDouble datamin_;    // the out data min
  GuiDouble datamax_;    // the out data max
  GuiString dataat_;     // the out data at
  GuiDouble minx_;       // the out bounding box
  GuiDouble miny_;
  GuiDouble minz_;
  GuiDouble maxx_;
  GuiDouble maxy_;
  GuiDouble maxz_;

  GuiInt cfldname_;      // change name
  GuiInt ctypename_;     // change type
  GuiInt cdataat_;       // change data at
  GuiInt cdataminmax_;   // change data value extents
  GuiInt cbbox_;         // change bbox

  CrowdMonitor widget_lock_;
  ScaledBoxWidget *box_;

  bool firsttime_;
  int widgetid_;
  pair<double,double> minmax_;

  EditField(const string& id);

  virtual ~EditField();

  template <class field_type_in, class field_type_out>
  field_type_out *create_edited_field(field_type_in *, field_type_out *);
  void clear_vals();
  void update_input_attributes(FieldHandle);
  bool check_types(FieldHandle);
  void build_widget(FieldHandle);

  virtual void execute();

  virtual void tcl_command(TCLArgs&, void*);
  virtual void widget_moved(int);
};

extern "C" PSECORESHARE Module* make_EditField(const string& id) {
  return scinew EditField(id);
}

EditField::EditField(const string& id)
  : Module("EditField", id, Source, "Fields", "SCIRun"),
    numnodes_("numnodes2", id, this),
    fldname_("fldname2", id, this),
    typename_("typename2", id, this),
    datamin_("datamin2", id, this),
    datamax_("datamax2", id, this),
    dataat_("dataat2", id, this),
    minx_("minx2", id, this),
    miny_("miny2", id, this),
    minz_("minz2", id, this),
    maxx_("maxx2", id, this),
    maxy_("maxy2", id, this),
    maxz_("maxz2", id, this),
    cfldname_("cfldname", id, this),
    ctypename_("ctypename", id, this),
    cdataat_("cdataat", id, this),
    cdataminmax_("cdataminmax", id, this),
    cbbox_("cbbox", id, this),
    widget_lock_("EditField widget lock"),
    minmax_(1,0)
  
{
  box_ = 0;
  firsttime_ = 1;
  widgetid_ = 0;
}

EditField::~EditField(){
}

double mag_val(double v) { return v; }
double mag_val(Vector v) { return v.length(); }


void EditField::clear_vals() 
{
  TCL::execute(string("set ")+id+"-fldname \"---\"");
  TCL::execute(string("set ")+id+"-typename \"---\"");
  TCL::execute(string("set ")+id+"-datamin \"---\"");
  TCL::execute(string("set ")+id+"-datamax \"---\"");
  TCL::execute(string("set ")+id+"-numnodes \"---\"");
  TCL::execute(string("set ")+id+"-numelems \"---\"");
  TCL::execute(string("set ")+id+"-dataat \"---\"");
  TCL::execute(string("set ")+id+"-minx \"---\"");
  TCL::execute(string("set ")+id+"-miny \"---\"");
  TCL::execute(string("set ")+id+"-minz \"---\"");
  TCL::execute(string("set ")+id+"-maxx \"---\"");
  TCL::execute(string("set ")+id+"-maxy \"---\"");
  TCL::execute(string("set ")+id+"-maxz \"---\"");
  TCL::execute(id+" update_multifields");
}

void EditField::update_input_attributes(FieldHandle f) 
{
  const string &tname = f->get_type_description()->get_name();
  TCL::execute(string("set ")+id+"-typename " + tname);

  switch(f->data_at()) {
  case Field::CELL: 
    TCL::execute(string("set ")+id+"-dataat Field::CELL"); break;
  case Field::NODE: 
    TCL::execute(string("set ")+id+"-dataat Field::NODE"); break;
  case Field::NONE: 
    TCL::execute(string("set ")+id+"-dataat Field::NONE"); break;
  default: ;
  }

  const TypeDescription *meshtd = f->mesh()->get_type_description();
  CompileInfo *ci = EditFieldAlgoCN::get_compile_info(meshtd);
  DynamicAlgoHandle algo_handle;
  if (! DynamicLoader::scirun_loader().get(*ci, algo_handle))
  {
    cout << "Could not compile algorithm." << std::endl;
    return;
  }
  EditFieldAlgoCN *algo =
    dynamic_cast<EditFieldAlgoCN *>(algo_handle.get_rep());
  if (algo == 0)
  {
    cout << "Could not get algorithm." << std::endl;
    return;
  }
  int num_nodes;
  int num_elems;
  algo->execute(f->mesh(), num_nodes, num_elems);

  TCL::execute(string("set ")+id+"-numnodes "+to_string(num_nodes));
  TCL::execute(string("set ")+id+"-numelems "+to_string(num_elems));


  const BBox bbox = f->mesh()->get_bounding_box();
  Point min = bbox.min();
  Point max = bbox.max();

  TCL::execute(string("set ")+id+"-minx "+to_string(min.x()));
  TCL::execute(string("set ")+id+"-miny "+to_string(min.y()));
  TCL::execute(string("set ")+id+"-minz "+to_string(min.z()));
  TCL::execute(string("set ")+id+"-maxx "+to_string(max.x()));
  TCL::execute(string("set ")+id+"-maxy "+to_string(max.y()));
  TCL::execute(string("set ")+id+"-maxz "+to_string(max.z()));

  ScalarFieldInterface *sdi = f->query_scalar_interface();
  if (sdi && f->data_at()!=Field::NONE) {
    sdi->compute_min_max(minmax_.first,minmax_.second);
    TCL::execute(string("set ")+id+"-datamin "+to_string(minmax_.first));
    TCL::execute(string("set ")+id+"-datamax "+to_string(minmax_.second));
  } else {
    TCL::execute(string("set ")+id+"-datamin \"--- N/A ---\"");
    TCL::execute(string("set ")+id+"-datamax \"--- N/A ---\"");
  }

  string fldname;
  if (f->get("name",fldname))
    TCL::execute(string("set ")+id+"-fldname "+fldname);
  else
    TCL::execute(string("set ")+id+"-fldname \"--- Name Not Assigned ---\"");

  TCL::execute(id+" update_multifields");
}

bool EditField::check_types(FieldHandle f)
{
  string fldtype = typename_.get();
  fldtype = fldtype.substr(0,fldtype.find('<'));
  if (fldtype!=f->get_type_name(0)) {
    postMessage(string("EditField: type mismatch ")+fldtype+", "+
		f->get_type_name(0));
    return false;
  } 

  int idx1,idx2;
  fldtype = typename_.get();
  idx1 = fldtype.find('<') + 1;
  idx2 = fldtype.find('>');
  fldtype = fldtype.substr(idx1,idx2-idx1);
  
  if ( (f->get_type_name(1) == "Vector" && fldtype != "Vector") ||
       (f->get_type_name(1) == "Tensor" && fldtype != "Tensor") ) {
    postMessage(string("EditField: type mismatch ")+fldtype+", "+
		f->get_type_name(1));
    return false;
  }

  return true;
}

template <class field_type_in, class field_type_out>
field_type_out *
EditField::create_edited_field(field_type_in *f, field_type_out * ) 
{
  typedef typename field_type_in::mesh_type          mesh_type_in;
  typedef typename field_type_in::fdata_type         fdata_type_in;
  typedef typename fdata_type_in::value_type         value_type_in;
  typedef typename field_type_out::mesh_type         mesh_type_out;
  typedef typename field_type_out::fdata_type        fdata_type_out;
  typedef typename fdata_type_out::value_type        value_type_out;
  typedef typename mesh_type_in::Node::iterator      node_iter;
  typedef typename mesh_type_in::Cell::iterator      cell_iter;
  typedef typename mesh_type_in::Elem::iterator      elem_iter;

  mesh_type_in *imesh = f->get_typed_mesh().get_rep(); // input mesh

  // create storage for a new mesh
  mesh_type_out *omesh = scinew mesh_type_out(*imesh);

  // transform the mesh if necessary
  if (cbbox_.get()) {
    BBox old = imesh->get_bounding_box();
    Point oldc = old.min()+(old.max()-old.min())/2.;
    Point center, right, down, in;
    box_->GetPosition(center,right,down,in);
    // rotate * scale * translate
    Transform t,r;
    Point unused;
    t.load_identity();
    r.load_frame(unused,(right-center).normal(),
		 (down-center).normal(),
		 (in-center).normal());
    t.pre_trans(r);
    t.pre_scale(Vector((right-center).length()/(old.max().x()-oldc.x()),
		       (down-center).length()/(old.max().y()-oldc.y()),
		       (in-center).length()/(old.max().z()-oldc.z())));
    t.pre_translate(Vector(center.x(),center.y(),center.z()));
    omesh->transform(t);
  }

  // identify the new data location
  Field::data_location dataat;
  if (cdataat_.get()) {
    string d = dataat_.get();
    if (d=="Field::CELL")
      dataat = Field::CELL;
    else if (d=="Field::NODE")
      dataat = Field::NODE;
    else 
      dataat = Field::NONE; // defaults to NONE
  } else
    dataat = f->data_at();

  // create the field with the new mesh and data location
  field_type_out *field = scinew field_type_out(omesh,dataat);

  // copy the (possibly transformed) data to the new field
  field->resize_fdata();
  fdata_type_out &ofdata = field->fdata();
  fdata_type_in &ifdata = f->fdata();
  TCL::execute(id + " set_state Executing 0");
  int loop = 0;
  typename fdata_type_in::iterator in = ifdata.begin();
  typename fdata_type_out::iterator out = ofdata.begin();
  if (dataat == f->data_at()) {
    if (!cdataminmax_.get()) {
      while (in != ifdata.end()) {
	*out = (value_type_out)(*in);
	++in; ++out; ++loop;
      }
    } else {
      while (in != ifdata.end()) {
	//linearly transform the data
	*out = (value_type_out)
	  (datamin_.get()+mag_val((((*in)-minmax_.first)/
				   (minmax_.second-minmax_.first))*
	   (datamax_.get()-datamin_.get())));
	++in; ++out; ++loop;
      }
    } 
  } else {
    // changing from node to cell or cell to node - now what?
  }

  // set some field attributes
  ScalarFieldInterface* sfi = field->query_scalar_interface();
  if (cfldname_.get()) 
    field->store(string("name"),fldname_.get());
  if (sfi) {
    std::pair<double,double> minmax(1,0);
    sfi->compute_min_max(minmax.first,minmax.second);
    field->store(string("minmax"),minmax);
  }
    
  return field;
}

void EditField::build_widget(FieldHandle f)
{
  double l2norm;
  Point center, right, down, in;
  Point min,max;

  if (!cbbox_.get()) {
    // build a widget identical to the BBox
    const BBox bbox = f->mesh()->get_bounding_box();
    min = bbox.min();
    max = bbox.max();
  } else {
    // build a widget as described by the UI
    min = Point(minx_.get(),miny_.get(),minz_.get());
    max = Point(maxx_.get(),maxy_.get(),maxz_.get());
  }
    
  center = Point(min.x()+(max.x()-min.x())/2.,
		 min.y()+(max.y()-min.y())/2.,
		 min.z()+(max.z()-min.z())/2.);
  right = center + Vector((max.x()-min.x())/2.,0,0);
  down = center + Vector(0,(max.y()-min.y())/2.,0);
  in = center + Vector(0,0,(max.z()-min.z())/2.);

  l2norm = (max-min).length();
  
  box_ = scinew ScaledBoxWidget(this,&widget_lock_,1);
  box_->SetScale(l2norm*.015);
  box_->SetPosition(center,right,down,in);
  box_->AxisAligned(1);
  
  GeomGroup *widget_group = scinew GeomGroup;
  widget_group->add(box_->GetWidget());

  GeometryOPort *ogport=0;
  ogport = (GeometryOPort*)get_oport(1);
  widgetid_ = ogport->addObj(widget_group,"EditField Transform widget",
			     &widget_lock_);
  ogport->flushViews();
}


void EditField::execute(){
  FieldIPort *iport=0; 
  FieldHandle fh;
  Field *f=0;
  FieldOPort *oport=0;
  
  // the output port is required
  if (!(oport=(FieldOPort*)get_oport(0)))
    return;

  // the input port (with data) is required
  if (!(iport=(FieldIPort*)get_iport(0)) || 
      !iport->get(fh) || 
      !(f=fh.get_rep())) {
    clear_vals();
    return;
  }

  // get and display the attributes of the input field
  update_input_attributes(fh);

  // build the transform widget
  if (firsttime_) {
    firsttime_ = false;
    build_widget(f);
  } 

  if (!cfldname_.get() &&
      !ctypename_.get() &&
      !cdataminmax_.get() &&
      !cdataat_.get() &&
      !cbbox_.get()) {
    // no changes, just send the original through (it may be nothing!)
    oport->send(f);    
    return;
  }

  // verify that the requested edits are possible (type check)
  if (ctypename_.get() && !check_types(fh))
    return;

  // create a field identical to the input, except for the edits
  string tn1 = f->get_type_name(-1);
  string tn2 = typename_.get(); 
  int ctype = ctypename_.get();
  Field *ef = 0;

  // oh man, is this a pain...
  if (!ctype) {
    if (tn1 == "TetVol<char>") {
      ef = create_edited_field((TetVol<char>*)f,(TetVol<char>*)0);
    } else if (tn1 == "TetVol<unsigned char>") {
      ef = create_edited_field((TetVol<unsigned char>*)f,
			       (TetVol<unsigned char>*)0);
    } else if (tn1 == "TetVol<short>") {
      ef = create_edited_field((TetVol<short>*)f,(TetVol<short>*)0);
    } else if (tn1 == "TetVol<unsigned short>") {
      ef = create_edited_field((TetVol<unsigned short>*)f,
			       (TetVol<unsigned short>*)0);
    } else if (tn1 == "TetVol<int>") {
      ef = create_edited_field((TetVol<int>*)f,(TetVol<int>*)0);
    } else if (tn1 == "TetVol<unsigned int>") {
      ef = create_edited_field((TetVol<unsigned int>*)f,
			       (TetVol<unsigned int>*)0);
    } else if (tn1 == "TetVol<float>") {
      ef = create_edited_field((TetVol<float>*)f,(TetVol<float>*)0);
    } else if (tn1 == "TetVol<double>") {
      ef = create_edited_field((TetVol<double>*)f,(TetVol<double>*)0);
    } else if (tn1 == "TetVol<long>") {
      ef = create_edited_field((TetVol<long>*)f,(TetVol<long>*)0);
    } else if (tn1 == "TetVol<Vector>") {
      ef = create_edited_field((TetVol<Vector>*)f,(TetVol<Vector>*)0);
      /*} else if (tn1 == "TetVol<Tensor>") {
	ef = create_edited_field((TetVol<Tensor>*)f,(TetVol<Tensor>*)0);*/
    } else if (tn1 == "TriSurf<char>") { 
      ef = create_edited_field((TriSurf<char>*)f,(TriSurf<char>*)0);
    } else if (tn1 == "TriSurf<unsigned char>") {
      ef = create_edited_field((TriSurf<unsigned char>*)f,
			       (TriSurf<unsigned char>*)0);
    } else if (tn1 == "TriSurf<short>") {
      ef = create_edited_field((TriSurf<short>*)f,(TriSurf<short>*)0);
    } else if (tn1 == "TriSurf<unsigned short>") {
      ef = create_edited_field((TriSurf<unsigned short>*)f,
			       (TriSurf<unsigned short>*)0);
    } else if (tn1 == "TriSurf<int>") {
      ef = create_edited_field((TriSurf<int>*)f,(TriSurf<int>*)0);
    } else if (tn1 == "TriSurf<unsigned int>") {
      ef = create_edited_field((TriSurf<unsigned int>*)f,
			       (TriSurf<unsigned int>*)0);
    } else if (tn1 == "TriSurf<float>") {
      ef = create_edited_field((TriSurf<float>*)f,(TriSurf<float>*)0);
    } else if (tn1 == "TriSurf<double>") {
      ef = create_edited_field((TriSurf<double>*)f,(TriSurf<double>*)0);
    } else if (tn1 == "TriSurf<long>") {
      ef = create_edited_field((TriSurf<long>*)f,(TriSurf<long>*)0);
    } else if (tn1 == "TriSurf<Vector>") {
      ef = create_edited_field((TriSurf<Vector>*)f,(TriSurf<Vector>*)0);
      /*} else if (tn1 == "TriSruf<Tensor>") {
	ef = create_edited_field((TriSurf<Tensor>*)f,(TriSurf<Tensor>*)0);*/
    } else if (tn1 == "LatticeVol<char>") {
      ef = create_edited_field((LatticeVol<char>*)f,(LatticeVol<char>*)0);
    } else if (tn1 == "LatticeVol<unsigned char>") {
      ef = create_edited_field((LatticeVol<unsigned char>*)f,
			       (LatticeVol<unsigned char>*)0);
    } else if (tn1 == "LatticeVol<short>") {
      ef = create_edited_field((LatticeVol<short>*)f,
			       (LatticeVol<short>*)0);
    } else if (tn1 == "LatticeVol<unsigned short>") {
      ef = create_edited_field((LatticeVol<unsigned short>*)f,
			       (LatticeVol<unsigned short>*)0);
    } else if (tn1 == "LatticeVol<int>") {
      ef = create_edited_field((LatticeVol<int>*)f,(LatticeVol<int>*)0);
    } else if (tn1 == "LatticeVol<unsigned int>") {
      ef = create_edited_field((LatticeVol<unsigned int>*)f,
			       (LatticeVol<unsigned int>*)0);
    } else if (tn1 == "LatticeVol<float>") {
      ef = create_edited_field((LatticeVol<float>*)f,(LatticeVol<float>*)0);
    } else if (tn1 == "LatticeVol<double>") {
      ef = create_edited_field((LatticeVol<double>*)f,(LatticeVol<double>*)0);
    } else if (tn1 == "LatticeVol<long>") {
      ef = create_edited_field((LatticeVol<long>*)f,(LatticeVol<long>*)0);
    } else if (tn1 == "LatticeVol<Vector>") {
      ef = create_edited_field((LatticeVol<Vector>*)f,(LatticeVol<Vector>*)0);
      /*} else if (tn1 == "LatticeVol<Tensor>") {
	ef = create_edited_field((LatticeVol<Tensor>*)f,(LatticeVol<Tensor>*)0);*/
    }
  } else {
    if (tn1 == "TetVol<char>") {
      if (tn2 == "TetVol<char>") {
	ef = create_edited_field((TetVol<char>*)f,(TetVol<char>*)0);
      } else if (tn2 == "TetVol<unsigned char>") {
	ef = create_edited_field((TetVol<char>*)f,
				 (TetVol<unsigned char>*)0);
      } else if (tn2 == "TetVol<short>") {
	ef = create_edited_field((TetVol<char>*)f,(TetVol<short>*)0);
      } else if (tn2 == "TetVol<unsigned short>") {
	ef = create_edited_field((TetVol<char>*)f,
				 (TetVol<unsigned short>*)0);
      } else if (tn2 == "TetVol<int>") {
	ef = create_edited_field((TetVol<char>*)f,(TetVol<int>*)0);
      } else if (tn2 == "TetVol<unsigned int>") {
	ef = create_edited_field((TetVol<char>*)f,
				 (TetVol<unsigned int>*)0);
      } else if (tn2 == "TetVol<float>") {
	ef = create_edited_field((TetVol<char>*)f,(TetVol<float>*)0);
      } else if (tn2 == "TetVol<double>") {
	ef = create_edited_field((TetVol<char>*)f,(TetVol<double>*)0);
      } else if (tn2 == "TetVol<long>") {
	ef = create_edited_field((TetVol<char>*)f,(TetVol<long>*)0);
      }
    } else if (tn1 == "TetVol<unsigned char>") {
      if (tn2 == "TetVol<char>") {
	ef = create_edited_field((TetVol<unsigned char>*)f,(TetVol<char>*)0);
      } else if (tn2 == "TetVol<unsigned char>") {
	ef = create_edited_field((TetVol<unsigned char>*)f,
				 (TetVol<unsigned char>*)0);
      } else if (tn2 == "TetVol<short>") {
	ef = create_edited_field((TetVol<unsigned char>*)f,(TetVol<short>*)0);
      } else if (tn2 == "TetVol<unsigned short>") {
	ef = create_edited_field((TetVol<unsigned char>*)f,
				 (TetVol<unsigned short>*)0);
      } else if (tn2 == "TetVol<int>") {
	ef = create_edited_field((TetVol<unsigned char>*)f,(TetVol<int>*)0);
      } else if (tn2 == "TetVol<unsigned int>") {
	ef = create_edited_field((TetVol<unsigned char>*)f,
				 (TetVol<unsigned int>*)0);
      } else if (tn2 == "TetVol<float>") {
	ef = create_edited_field((TetVol<unsigned char>*)f,(TetVol<float>*)0);
      } else if (tn2 == "TetVol<double>") {
	ef = create_edited_field((TetVol<unsigned char>*)f,(TetVol<double>*)0);
      } else if (tn2 == "TetVol<long>") {
	ef = create_edited_field((TetVol<unsigned char>*)f,(TetVol<long>*)0);
      }
    } else if (tn1 == "TetVol<short>") {
      if (tn2 == "TetVol<char>") {
	ef = create_edited_field((TetVol<short>*)f,(TetVol<char>*)0);
      } else if (tn2 == "TetVol<unsigned char>") {
	ef = create_edited_field((TetVol<short>*)f,
				 (TetVol<unsigned char>*)0);
      } else if (tn2 == "TetVol<short>") {
	ef = create_edited_field((TetVol<short>*)f,(TetVol<short>*)0);
      } else if (tn2 == "TetVol<unsigned short>") {
	ef = create_edited_field((TetVol<short>*)f,
				 (TetVol<unsigned short>*)0);
      } else if (tn2 == "TetVol<int>") {
	ef = create_edited_field((TetVol<short>*)f,(TetVol<int>*)0);
      } else if (tn2 == "TetVol<unsigned int>") {
	ef = create_edited_field((TetVol<short>*)f,
				 (TetVol<unsigned int>*)0);
      } else if (tn2 == "TetVol<float>") {
	ef = create_edited_field((TetVol<short>*)f,(TetVol<float>*)0);
      } else if (tn2 == "TetVol<double>") {
	ef = create_edited_field((TetVol<short>*)f,(TetVol<double>*)0);
      } else if (tn2 == "TetVol<long>") {
	ef = create_edited_field((TetVol<short>*)f,(TetVol<long>*)0);
      }
    } else if (tn1 == "TetVol<unsigned short>") {
      if (tn2 == "TetVol<char>") {
	ef = create_edited_field((TetVol<unsigned short>*)f,(TetVol<char>*)0);
      } else if (tn2 == "TetVol<unsigned char>") {
	ef = create_edited_field((TetVol<unsigned short>*)f,
				 (TetVol<unsigned char>*)0);
      } else if (tn2 == "TetVol<short>") {
	ef = create_edited_field((TetVol<unsigned short>*)f,(TetVol<short>*)0);
      } else if (tn2 == "TetVol<unsigned short>") {
	ef = create_edited_field((TetVol<unsigned short>*)f,
				 (TetVol<unsigned short>*)0);
      } else if (tn2 == "TetVol<int>") {
	ef = create_edited_field((TetVol<unsigned short>*)f,(TetVol<int>*)0);
      } else if (tn2 == "TetVol<unsigned int>") {
	ef = create_edited_field((TetVol<unsigned short>*)f,
				 (TetVol<unsigned int>*)0);
      } else if (tn2 == "TetVol<float>") {
	ef = create_edited_field((TetVol<unsigned short>*)f,(TetVol<float>*)0);
      } else if (tn2 == "TetVol<double>") {
	ef = create_edited_field((TetVol<unsigned short>*)f,(TetVol<double>*)0);
      } else if (tn2 == "TetVol<long>") {
	ef = create_edited_field((TetVol<unsigned short>*)f,(TetVol<long>*)0);
      }
    } else if (tn1 == "TetVol<int>") {
      if (tn2 == "TetVol<char>") {
	ef = create_edited_field((TetVol<unsigned short>*)f,(TetVol<char>*)0);
      } else if (tn2 == "TetVol<unsigned char>") {
	ef = create_edited_field((TetVol<unsigned short>*)f,
				 (TetVol<unsigned char>*)0);
      } else if (tn2 == "TetVol<short>") {
	ef = create_edited_field((TetVol<unsigned short>*)f,(TetVol<short>*)0);
      } else if (tn2 == "TetVol<unsigned short>") {
	ef = create_edited_field((TetVol<unsigned short>*)f,
				 (TetVol<unsigned short>*)0);
      } else if (tn2 == "TetVol<int>") {
	ef = create_edited_field((TetVol<unsigned short>*)f,(TetVol<int>*)0);
      } else if (tn2 == "TetVol<unsigned int>") {
	ef = create_edited_field((TetVol<unsigned short>*)f,
				 (TetVol<unsigned int>*)0);
      } else if (tn2 == "TetVol<float>") {
	ef = create_edited_field((TetVol<unsigned short>*)f,(TetVol<float>*)0);
      } else if (tn2 == "TetVol<double>") {
	ef = create_edited_field((TetVol<unsigned short>*)f,(TetVol<double>*)0);
      } else if (tn2 == "TetVol<long>") {
	ef = create_edited_field((TetVol<unsigned short>*)f,(TetVol<long>*)0);
      }
    } else if (tn1 == "TetVol<unsigned int>") {
      if (tn2 == "TetVol<char>") {
	ef = create_edited_field((TetVol<unsigned int>*)f,(TetVol<char>*)0);
      } else if (tn2 == "TetVol<unsigned char>") {
	ef = create_edited_field((TetVol<unsigned int>*)f,
				 (TetVol<unsigned char>*)0);
      } else if (tn2 == "TetVol<short>") {
	ef = create_edited_field((TetVol<unsigned int>*)f,(TetVol<short>*)0);
      } else if (tn2 == "TetVol<unsigned short>") {
	ef = create_edited_field((TetVol<unsigned int>*)f,
				 (TetVol<unsigned short>*)0);
      } else if (tn2 == "TetVol<int>") {
	ef = create_edited_field((TetVol<unsigned int>*)f,(TetVol<int>*)0);
      } else if (tn2 == "TetVol<unsigned int>") {
	ef = create_edited_field((TetVol<unsigned int>*)f,
				 (TetVol<unsigned int>*)0);
      } else if (tn2 == "TetVol<float>") {
	ef = create_edited_field((TetVol<unsigned int>*)f,(TetVol<float>*)0);
      } else if (tn2 == "TetVol<double>") {
	ef = create_edited_field((TetVol<unsigned int>*)f,(TetVol<double>*)0);
      } else if (tn2 == "TetVol<long>") {
	ef = create_edited_field((TetVol<unsigned int>*)f,(TetVol<long>*)0);
      }
    } else if (tn1 == "TetVol<float>") {
      if (tn2 == "TetVol<char>") {
	ef = create_edited_field((TetVol<float>*)f,(TetVol<char>*)0);
      } else if (tn2 == "TetVol<unsigned char>") {
	ef = create_edited_field((TetVol<float>*)f,
				 (TetVol<unsigned char>*)0);
      } else if (tn2 == "TetVol<short>") {
	ef = create_edited_field((TetVol<float>*)f,(TetVol<short>*)0);
      } else if (tn2 == "TetVol<unsigned short>") {
	ef = create_edited_field((TetVol<float>*)f,
				 (TetVol<unsigned short>*)0);
      } else if (tn2 == "TetVol<int>") {
	ef = create_edited_field((TetVol<float>*)f,(TetVol<int>*)0);
      } else if (tn2 == "TetVol<unsigned int>") {
	ef = create_edited_field((TetVol<float>*)f,
				 (TetVol<unsigned int>*)0);
      } else if (tn2 == "TetVol<float>") {
	ef = create_edited_field((TetVol<float>*)f,(TetVol<float>*)0);
      } else if (tn2 == "TetVol<double>") {
	ef = create_edited_field((TetVol<float>*)f,(TetVol<double>*)0);
      } else if (tn2 == "TetVol<long>") {
	ef = create_edited_field((TetVol<float>*)f,(TetVol<long>*)0);
      }
    } else if (tn1 == "TetVol<double>") {
      if (tn2 == "TetVol<char>") {
	ef = create_edited_field((TetVol<double>*)f,(TetVol<char>*)0);
      } else if (tn2 == "TetVol<unsigned char>") {
	ef = create_edited_field((TetVol<double>*)f,
				 (TetVol<unsigned char>*)0);
      } else if (tn2 == "TetVol<short>") {
	ef = create_edited_field((TetVol<double>*)f,(TetVol<short>*)0);
      } else if (tn2 == "TetVol<unsigned short>") {
	ef = create_edited_field((TetVol<double>*)f,
				 (TetVol<unsigned short>*)0);
      } else if (tn2 == "TetVol<int>") {
	ef = create_edited_field((TetVol<double>*)f,(TetVol<int>*)0);
      } else if (tn2 == "TetVol<unsigned int>") {
	ef = create_edited_field((TetVol<double>*)f,
				 (TetVol<unsigned int>*)0);
      } else if (tn2 == "TetVol<float>") {
	ef = create_edited_field((TetVol<double>*)f,(TetVol<float>*)0);
      } else if (tn2 == "TetVol<double>") {
	ef = create_edited_field((TetVol<double>*)f,(TetVol<double>*)0);
      } else if (tn2 == "TetVol<long>") {
	ef = create_edited_field((TetVol<double>*)f,(TetVol<long>*)0);
      }
    } else if (tn1 == "TetVol<long>") {
      if (tn2 == "TetVol<char>") {
	ef = create_edited_field((TetVol<long>*)f,(TetVol<char>*)0);
      } else if (tn2 == "TetVol<unsigned char>") {
	ef = create_edited_field((TetVol<long>*)f,
				 (TetVol<unsigned char>*)0);
      } else if (tn2 == "TetVol<short>") {
	ef = create_edited_field((TetVol<long>*)f,(TetVol<short>*)0);
      } else if (tn2 == "TetVol<unsigned short>") {
	ef = create_edited_field((TetVol<long>*)f,
				 (TetVol<unsigned short>*)0);
      } else if (tn2 == "TetVol<int>") {
	ef = create_edited_field((TetVol<long>*)f,(TetVol<int>*)0);
      } else if (tn2 == "TetVol<unsigned int>") {
	ef = create_edited_field((TetVol<long>*)f,
				 (TetVol<unsigned int>*)0);
      } else if (tn2 == "TetVol<float>") {
	ef = create_edited_field((TetVol<long>*)f,(TetVol<float>*)0);
      } else if (tn2 == "TetVol<double>") {
	ef = create_edited_field((TetVol<long>*)f,(TetVol<double>*)0);
      } else if (tn2 == "TetVol<long>") {
	ef = create_edited_field((TetVol<long>*)f,(TetVol<long>*)0);
      }
    }

    if (tn1 == "TriSurf<char>") {
      if (tn2 == "TriSurf<char>") {
	ef = create_edited_field((TriSurf<char>*)f,(TriSurf<char>*)0);
      } else if (tn2 == "TriSurf<unsigned char>") {
	ef = create_edited_field((TriSurf<char>*)f,
				 (TriSurf<unsigned char>*)0);
      } else if (tn2 == "TriSurf<short>") {
	ef = create_edited_field((TriSurf<char>*)f,(TriSurf<short>*)0);
      } else if (tn2 == "TriSurf<unsigned short>") {
	ef = create_edited_field((TriSurf<char>*)f,
				 (TriSurf<unsigned short>*)0);
      } else if (tn2 == "TriSurf<int>") {
	ef = create_edited_field((TriSurf<char>*)f,(TriSurf<int>*)0);
      } else if (tn2 == "TriSurf<unsigned int>") {
	ef = create_edited_field((TriSurf<char>*)f,
				 (TriSurf<unsigned int>*)0);
      } else if (tn2 == "TriSurf<float>") {
	ef = create_edited_field((TriSurf<char>*)f,(TriSurf<float>*)0);
      } else if (tn2 == "TriSurf<double>") {
	ef = create_edited_field((TriSurf<char>*)f,(TriSurf<double>*)0);
      } else if (tn2 == "TriSurf<long>") {
	ef = create_edited_field((TriSurf<char>*)f,(TriSurf<long>*)0);
      }
    } else if (tn1 == "TriSurf<unsigned char>") {
      if (tn2 == "TriSurf<char>") {
	ef = create_edited_field((TriSurf<unsigned char>*)f,(TriSurf<char>*)0);
      } else if (tn2 == "TriSurf<unsigned char>") {
	ef = create_edited_field((TriSurf<unsigned char>*)f,
				 (TriSurf<unsigned char>*)0);
      } else if (tn2 == "TriSurf<short>") {
	ef = create_edited_field((TriSurf<unsigned char>*)f,(TriSurf<short>*)0);
      } else if (tn2 == "TriSurf<unsigned short>") {
	ef = create_edited_field((TriSurf<unsigned char>*)f,
				 (TriSurf<unsigned short>*)0);
      } else if (tn2 == "TriSurf<int>") {
	ef = create_edited_field((TriSurf<unsigned char>*)f,(TriSurf<int>*)0);
      } else if (tn2 == "TriSurf<unsigned int>") {
	ef = create_edited_field((TriSurf<unsigned char>*)f,
				 (TriSurf<unsigned int>*)0);
      } else if (tn2 == "TriSurf<float>") {
	ef = create_edited_field((TriSurf<unsigned char>*)f,(TriSurf<float>*)0);
      } else if (tn2 == "TriSurf<double>") {
	ef = create_edited_field((TriSurf<unsigned char>*)f,(TriSurf<double>*)0);
      } else if (tn2 == "TriSurf<long>") {
	ef = create_edited_field((TriSurf<unsigned char>*)f,(TriSurf<long>*)0);
      }
    } else if (tn1 == "TriSurf<short>") {
      if (tn2 == "TriSurf<char>") {
	ef = create_edited_field((TriSurf<short>*)f,(TriSurf<char>*)0);
      } else if (tn2 == "TriSurf<unsigned char>") {
	ef = create_edited_field((TriSurf<short>*)f,
				 (TriSurf<unsigned char>*)0);
      } else if (tn2 == "TriSurf<short>") {
	ef = create_edited_field((TriSurf<short>*)f,(TriSurf<short>*)0);
      } else if (tn2 == "TriSurf<unsigned short>") {
	ef = create_edited_field((TriSurf<short>*)f,
				 (TriSurf<unsigned short>*)0);
      } else if (tn2 == "TriSurf<int>") {
	ef = create_edited_field((TriSurf<short>*)f,(TriSurf<int>*)0);
      } else if (tn2 == "TriSurf<unsigned int>") {
	ef = create_edited_field((TriSurf<short>*)f,
				 (TriSurf<unsigned int>*)0);
      } else if (tn2 == "TriSurf<float>") {
	ef = create_edited_field((TriSurf<short>*)f,(TriSurf<float>*)0);
      } else if (tn2 == "TriSurf<double>") {
	ef = create_edited_field((TriSurf<short>*)f,(TriSurf<double>*)0);
      } else if (tn2 == "TriSurf<long>") {
	ef = create_edited_field((TriSurf<short>*)f,(TriSurf<long>*)0);
      }
    } else if (tn1 == "TriSurf<unsigned short>") {
      if (tn2 == "TriSurf<char>") {
	ef = create_edited_field((TriSurf<unsigned short>*)f,(TriSurf<char>*)0);
      } else if (tn2 == "TriSurf<unsigned char>") {
	ef = create_edited_field((TriSurf<unsigned short>*)f,
				 (TriSurf<unsigned char>*)0);
      } else if (tn2 == "TriSurf<short>") {
	ef = create_edited_field((TriSurf<unsigned short>*)f,(TriSurf<short>*)0);
      } else if (tn2 == "TriSurf<unsigned short>") {
	ef = create_edited_field((TriSurf<unsigned short>*)f,
				 (TriSurf<unsigned short>*)0);
      } else if (tn2 == "TriSurf<int>") {
	ef = create_edited_field((TriSurf<unsigned short>*)f,(TriSurf<int>*)0);
      } else if (tn2 == "TriSurf<unsigned int>") {
	ef = create_edited_field((TriSurf<unsigned short>*)f,
				 (TriSurf<unsigned int>*)0);
      } else if (tn2 == "TriSurf<float>") {
	ef = create_edited_field((TriSurf<unsigned short>*)f,(TriSurf<float>*)0);
      } else if (tn2 == "TriSurf<double>") {
	ef = create_edited_field((TriSurf<unsigned short>*)f,(TriSurf<double>*)0);
      } else if (tn2 == "TriSurf<long>") {
	ef = create_edited_field((TriSurf<unsigned short>*)f,(TriSurf<long>*)0);
      }
    } else if (tn1 == "TriSurf<int>") {
      if (tn2 == "TriSurf<char>") {
	ef = create_edited_field((TriSurf<int>*)f,(TriSurf<char>*)0);
      } else if (tn2 == "TriSurf<unsigned char>") {
	ef = create_edited_field((TriSurf<int>*)f,
				 (TriSurf<unsigned char>*)0);
      } else if (tn2 == "TriSurf<short>") {
	ef = create_edited_field((TriSurf<int>*)f,(TriSurf<short>*)0);
      } else if (tn2 == "TriSurf<unsigned short>") {
	ef = create_edited_field((TriSurf<int>*)f,
				 (TriSurf<unsigned short>*)0);
      } else if (tn2 == "TriSurf<int>") {
	ef = create_edited_field((TriSurf<int>*)f,(TriSurf<int>*)0);
      } else if (tn2 == "TriSurf<unsigned int>") {
	ef = create_edited_field((TriSurf<int>*)f,
				 (TriSurf<unsigned int>*)0);
      } else if (tn2 == "TriSurf<float>") {
	ef = create_edited_field((TriSurf<int>*)f,(TriSurf<float>*)0);
      } else if (tn2 == "TriSurf<double>") {
	ef = create_edited_field((TriSurf<int>*)f,(TriSurf<double>*)0);
      } else if (tn2 == "TriSurf<long>") {
	ef = create_edited_field((TriSurf<int>*)f,(TriSurf<long>*)0);
      }
    } else if (tn1 == "TriSurf<unsigned int>") {
      if (tn2 == "TriSurf<char>") {
	ef = create_edited_field((TriSurf<unsigned int>*)f,(TriSurf<char>*)0);
      } else if (tn2 == "TriSurf<unsigned char>") {
	ef = create_edited_field((TriSurf<unsigned int>*)f,
				 (TriSurf<unsigned char>*)0);
      } else if (tn2 == "TriSurf<short>") {
	ef = create_edited_field((TriSurf<unsigned int>*)f,(TriSurf<short>*)0);
      } else if (tn2 == "TriSurf<unsigned short>") {
	ef = create_edited_field((TriSurf<unsigned int>*)f,
				 (TriSurf<unsigned short>*)0);
      } else if (tn2 == "TriSurf<int>") {
	ef = create_edited_field((TriSurf<unsigned int>*)f,(TriSurf<int>*)0);
      } else if (tn2 == "TriSurf<unsigned int>") {
	ef = create_edited_field((TriSurf<unsigned int>*)f,
				 (TriSurf<unsigned int>*)0);
      } else if (tn2 == "TriSurf<float>") {
	ef = create_edited_field((TriSurf<unsigned int>*)f,(TriSurf<float>*)0);
      } else if (tn2 == "TriSurf<double>") {
	ef = create_edited_field((TriSurf<unsigned int>*)f,(TriSurf<double>*)0);
      } else if (tn2 == "TriSurf<long>") {
	ef = create_edited_field((TriSurf<unsigned int>*)f,(TriSurf<long>*)0);
      }
    } else if (tn1 == "TriSurf<float>") {
      if (tn2 == "TriSurf<char>") {
	ef = create_edited_field((TriSurf<float>*)f,(TriSurf<char>*)0);
      } else if (tn2 == "TriSurf<unsigned char>") {
	ef = create_edited_field((TriSurf<float>*)f,
				 (TriSurf<unsigned char>*)0);
      } else if (tn2 == "TriSurf<short>") {
	ef = create_edited_field((TriSurf<float>*)f,(TriSurf<short>*)0);
      } else if (tn2 == "TriSurf<unsigned short>") {
	ef = create_edited_field((TriSurf<float>*)f,
				 (TriSurf<unsigned short>*)0);
      } else if (tn2 == "TriSurf<int>") {
	ef = create_edited_field((TriSurf<float>*)f,(TriSurf<int>*)0);
      } else if (tn2 == "TriSurf<unsigned int>") {
	ef = create_edited_field((TriSurf<float>*)f,
				 (TriSurf<unsigned int>*)0);
      } else if (tn2 == "TriSurf<float>") {
	ef = create_edited_field((TriSurf<float>*)f,(TriSurf<float>*)0);
      } else if (tn2 == "TriSurf<double>") {
	ef = create_edited_field((TriSurf<float>*)f,(TriSurf<double>*)0);
      } else if (tn2 == "TriSurf<long>") {
	ef = create_edited_field((TriSurf<float>*)f,(TriSurf<long>*)0);
      }
    } else if (tn1 == "TriSurf<double>") {
      if (tn2 == "TriSurf<char>") {
	ef = create_edited_field((TriSurf<double>*)f,(TriSurf<char>*)0);
      } else if (tn2 == "TriSurf<unsigned char>") {
	ef = create_edited_field((TriSurf<double>*)f,
				 (TriSurf<unsigned char>*)0);
      } else if (tn2 == "TriSurf<short>") {
	ef = create_edited_field((TriSurf<double>*)f,(TriSurf<short>*)0);
      } else if (tn2 == "TriSurf<unsigned short>") {
	ef = create_edited_field((TriSurf<double>*)f,
				 (TriSurf<unsigned short>*)0);
      } else if (tn2 == "TriSurf<int>") {
	ef = create_edited_field((TriSurf<double>*)f,(TriSurf<int>*)0);
      } else if (tn2 == "TriSurf<unsigned int>") {
	ef = create_edited_field((TriSurf<double>*)f,
				 (TriSurf<unsigned int>*)0);
      } else if (tn2 == "TriSurf<float>") {
	ef = create_edited_field((TriSurf<double>*)f,(TriSurf<float>*)0);
      } else if (tn2 == "TriSurf<double>") {
	ef = create_edited_field((TriSurf<double>*)f,(TriSurf<double>*)0);
      } else if (tn2 == "TriSurf<long>") {
	ef = create_edited_field((TriSurf<double>*)f,(TriSurf<long>*)0);
      }
    } else if (tn1 == "TriSurf<long>") {
      if (tn2 == "TriSurf<char>") {
	ef = create_edited_field((TriSurf<long>*)f,(TriSurf<char>*)0);
      } else if (tn2 == "TriSurf<unsigned char>") {
	ef = create_edited_field((TriSurf<long>*)f,
				 (TriSurf<unsigned char>*)0);
      } else if (tn2 == "TriSurf<short>") {
	ef = create_edited_field((TriSurf<long>*)f,(TriSurf<short>*)0);
      } else if (tn2 == "TriSurf<unsigned short>") {
	ef = create_edited_field((TriSurf<long>*)f,
				 (TriSurf<unsigned short>*)0);
      } else if (tn2 == "TriSurf<int>") {
	ef = create_edited_field((TriSurf<long>*)f,(TriSurf<int>*)0);
      } else if (tn2 == "TriSurf<unsigned int>") {
	ef = create_edited_field((TriSurf<long>*)f,
				 (TriSurf<unsigned int>*)0);
      } else if (tn2 == "TriSurf<float>") {
	ef = create_edited_field((TriSurf<long>*)f,(TriSurf<float>*)0);
      } else if (tn2 == "TriSurf<double>") {
	ef = create_edited_field((TriSurf<long>*)f,(TriSurf<double>*)0);
      } else if (tn2 == "TriSurf<long>") {
	ef = create_edited_field((TriSurf<long>*)f,(TriSurf<long>*)0);
      }
    }

    if (tn1 == "LatticeVol<char>") {
      if (tn2 == "LatticeVol<char>") {
	ef = create_edited_field((LatticeVol<char>*)f,(LatticeVol<char>*)0);
      } else if (tn2 == "LatticeVol<unsigned char>") {
	ef = create_edited_field((LatticeVol<char>*)f,
				 (LatticeVol<unsigned char>*)0);
      } else if (tn2 == "LatticeVol<short>") {
	ef = create_edited_field((LatticeVol<char>*)f,(LatticeVol<short>*)0);
      } else if (tn2 == "LatticeVol<unsigned short>") {
	ef = create_edited_field((LatticeVol<char>*)f,
				 (LatticeVol<unsigned short>*)0);
      } else if (tn2 == "LatticeVol<int>") {
	ef = create_edited_field((LatticeVol<char>*)f,(LatticeVol<int>*)0);
      } else if (tn2 == "LatticeVol<unsigned int>") {
	ef = create_edited_field((LatticeVol<char>*)f,
				 (LatticeVol<unsigned int>*)0);
      } else if (tn2 == "LatticeVol<float>") {
	ef = create_edited_field((LatticeVol<char>*)f,(LatticeVol<float>*)0);
      } else if (tn2 == "LatticeVol<double>") {
	ef = create_edited_field((LatticeVol<char>*)f,(LatticeVol<double>*)0);
      } else if (tn2 == "LatticeVol<long>") {
	ef = create_edited_field((LatticeVol<char>*)f,(LatticeVol<long>*)0);
      }
    } else if (tn1 == "LatticeVol<unsigned char>") {
      if (tn2 == "LatticeVol<char>") {
	ef = create_edited_field((LatticeVol<unsigned char>*)f,(LatticeVol<char>*)0);
      } else if (tn2 == "LatticeVol<unsigned char>") {
	ef = create_edited_field((LatticeVol<unsigned char>*)f,
				 (LatticeVol<unsigned char>*)0);
      } else if (tn2 == "LatticeVol<short>") {
	ef = create_edited_field((LatticeVol<unsigned char>*)f,(LatticeVol<short>*)0);
      } else if (tn2 == "LatticeVol<unsigned short>") {
	ef = create_edited_field((LatticeVol<unsigned char>*)f,
				 (LatticeVol<unsigned short>*)0);
      } else if (tn2 == "LatticeVol<int>") {
	ef = create_edited_field((LatticeVol<unsigned char>*)f,(LatticeVol<int>*)0);
      } else if (tn2 == "LatticeVol<unsigned int>") {
	ef = create_edited_field((LatticeVol<unsigned char>*)f,
				 (LatticeVol<unsigned int>*)0);
      } else if (tn2 == "LatticeVol<float>") {
	ef = create_edited_field((LatticeVol<unsigned char>*)f,(LatticeVol<float>*)0);
      } else if (tn2 == "LatticeVol<double>") {
	ef = create_edited_field((LatticeVol<unsigned char>*)f,(LatticeVol<double>*)0);
      } else if (tn2 == "LatticeVol<long>") {
	ef = create_edited_field((LatticeVol<unsigned char>*)f,(LatticeVol<long>*)0);
      }
    } else if (tn1 == "LatticeVol<short>") {
      if (tn2 == "LatticeVol<char>") {
	ef = create_edited_field((LatticeVol<short>*)f,(LatticeVol<char>*)0);
      } else if (tn2 == "LatticeVol<unsigned char>") {
	ef = create_edited_field((LatticeVol<short>*)f,
				 (LatticeVol<unsigned char>*)0);
      } else if (tn2 == "LatticeVol<short>") {
	ef = create_edited_field((LatticeVol<short>*)f,(LatticeVol<short>*)0);
      } else if (tn2 == "LatticeVol<unsigned short>") {
	ef = create_edited_field((LatticeVol<short>*)f,
				 (LatticeVol<unsigned short>*)0);
      } else if (tn2 == "LatticeVol<int>") {
	ef = create_edited_field((LatticeVol<short>*)f,(LatticeVol<int>*)0);
      } else if (tn2 == "LatticeVol<unsigned int>") {
	ef = create_edited_field((LatticeVol<short>*)f,
				 (LatticeVol<unsigned int>*)0);
      } else if (tn2 == "LatticeVol<float>") {
	ef = create_edited_field((LatticeVol<short>*)f,(LatticeVol<float>*)0);
      } else if (tn2 == "LatticeVol<double>") {
	ef = create_edited_field((LatticeVol<short>*)f,(LatticeVol<double>*)0);
      } else if (tn2 == "LatticeVol<long>") {
	ef = create_edited_field((LatticeVol<short>*)f,(LatticeVol<long>*)0);
      }
    } else if (tn1 == "LatticeVol<unsigned short>") {
      if (tn2 == "LatticeVol<char>") {
	ef = create_edited_field((LatticeVol<unsigned short>*)f,(LatticeVol<char>*)0);
      } else if (tn2 == "LatticeVol<unsigned char>") {
	ef = create_edited_field((LatticeVol<unsigned short>*)f,
				 (LatticeVol<unsigned char>*)0);
      } else if (tn2 == "LatticeVol<short>") {
	ef = create_edited_field((LatticeVol<unsigned short>*)f,(LatticeVol<short>*)0);
      } else if (tn2 == "LatticeVol<unsigned short>") {
	ef = create_edited_field((LatticeVol<unsigned short>*)f,
				 (LatticeVol<unsigned short>*)0);
      } else if (tn2 == "LatticeVol<int>") {
	ef = create_edited_field((LatticeVol<unsigned short>*)f,(LatticeVol<int>*)0);
      } else if (tn2 == "LatticeVol<unsigned int>") {
	ef = create_edited_field((LatticeVol<unsigned short>*)f,
				 (LatticeVol<unsigned int>*)0);
      } else if (tn2 == "LatticeVol<float>") {
	ef = create_edited_field((LatticeVol<unsigned short>*)f,(LatticeVol<float>*)0);
      } else if (tn2 == "LatticeVol<double>") {
	ef = create_edited_field((LatticeVol<unsigned short>*)f,(LatticeVol<double>*)0);
      } else if (tn2 == "LatticeVol<long>") {
	ef = create_edited_field((LatticeVol<unsigned short>*)f,(LatticeVol<long>*)0);
      }
    } else if (tn1 == "LatticeVol<int>") {
      if (tn2 == "LatticeVol<char>") {
	ef = create_edited_field((LatticeVol<char>*)f,(LatticeVol<char>*)0);
      } else if (tn2 == "LatticeVol<unsigned char>") {
	ef = create_edited_field((LatticeVol<char>*)f,
				 (LatticeVol<unsigned char>*)0);
      } else if (tn2 == "LatticeVol<short>") {
	ef = create_edited_field((LatticeVol<char>*)f,(LatticeVol<short>*)0);
      } else if (tn2 == "LatticeVol<unsigned short>") {
	ef = create_edited_field((LatticeVol<char>*)f,
				 (LatticeVol<unsigned short>*)0);
      } else if (tn2 == "LatticeVol<int>") {
	ef = create_edited_field((LatticeVol<char>*)f,(LatticeVol<int>*)0);
      } else if (tn2 == "LatticeVol<unsigned int>") {
	ef = create_edited_field((LatticeVol<char>*)f,
				 (LatticeVol<unsigned int>*)0);
      } else if (tn2 == "LatticeVol<float>") {
	ef = create_edited_field((LatticeVol<char>*)f,(LatticeVol<float>*)0);
      } else if (tn2 == "LatticeVol<double>") {
	ef = create_edited_field((LatticeVol<char>*)f,(LatticeVol<double>*)0);
      } else if (tn2 == "LatticeVol<long>") {
	ef = create_edited_field((LatticeVol<char>*)f,(LatticeVol<long>*)0);
      }
    } else if (tn1 == "LatticeVol<unsigned int>") {
      if (tn2 == "LatticeVol<char>") {
	ef = create_edited_field((LatticeVol<unsigned int>*)f,(LatticeVol<char>*)0);
      } else if (tn2 == "LatticeVol<unsigned char>") {
	ef = create_edited_field((LatticeVol<unsigned int>*)f,
				 (LatticeVol<unsigned char>*)0);
      } else if (tn2 == "LatticeVol<short>") {
	ef = create_edited_field((LatticeVol<unsigned int>*)f,(LatticeVol<short>*)0);
      } else if (tn2 == "LatticeVol<unsigned short>") {
	ef = create_edited_field((LatticeVol<unsigned int>*)f,
				 (LatticeVol<unsigned short>*)0);
      } else if (tn2 == "LatticeVol<int>") {
	ef = create_edited_field((LatticeVol<int>*)f,(LatticeVol<int>*)0);
      } else if (tn2 == "LatticeVol<unsigned int>") {
	ef = create_edited_field((LatticeVol<unsigned int>*)f,
				 (LatticeVol<unsigned int>*)0);
      } else if (tn2 == "LatticeVol<float>") {
	ef = create_edited_field((LatticeVol<unsigned int>*)f,(LatticeVol<float>*)0);
      } else if (tn2 == "LatticeVol<double>") {
	ef = create_edited_field((LatticeVol<unsigned int>*)f,(LatticeVol<double>*)0);
      } else if (tn2 == "LatticeVol<long>") {
	ef = create_edited_field((LatticeVol<unsigned int>*)f,(LatticeVol<long>*)0);
      }
    } else if (tn1 == "LatticeVol<float>") {
      if (tn2 == "LatticeVol<char>") {
	ef = create_edited_field((LatticeVol<float>*)f,(LatticeVol<char>*)0);
      } else if (tn2 == "LatticeVol<unsigned char>") {
	ef = create_edited_field((LatticeVol<float>*)f,
				 (LatticeVol<unsigned char>*)0);
      } else if (tn2 == "LatticeVol<short>") {
	ef = create_edited_field((LatticeVol<float>*)f,(LatticeVol<short>*)0);
      } else if (tn2 == "LatticeVol<unsigned short>") {
	ef = create_edited_field((LatticeVol<float>*)f,
				 (LatticeVol<unsigned short>*)0);
      } else if (tn2 == "LatticeVol<int>") {
	ef = create_edited_field((LatticeVol<float>*)f,(LatticeVol<int>*)0);
      } else if (tn2 == "LatticeVol<unsigned int>") {
	ef = create_edited_field((LatticeVol<float>*)f,
				 (LatticeVol<unsigned int>*)0);
      } else if (tn2 == "LatticeVol<float>") {
	ef = create_edited_field((LatticeVol<float>*)f,(LatticeVol<float>*)0);
      } else if (tn2 == "LatticeVol<double>") {
	ef = create_edited_field((LatticeVol<float>*)f,(LatticeVol<double>*)0);
      } else if (tn2 == "LatticeVol<long>") {
	ef = create_edited_field((LatticeVol<float>*)f,(LatticeVol<long>*)0);
      }
    } else if (tn1 == "LatticeVol<double>") {
      if (tn2 == "LatticeVol<char>") {
	ef = create_edited_field((LatticeVol<double>*)f,(LatticeVol<char>*)0);
      } else if (tn2 == "LatticeVol<unsigned char>") {
	ef = create_edited_field((LatticeVol<double>*)f,
				 (LatticeVol<unsigned char>*)0);
      } else if (tn2 == "LatticeVol<short>") {
	ef = create_edited_field((LatticeVol<double>*)f,(LatticeVol<short>*)0);
      } else if (tn2 == "LatticeVol<unsigned short>") {
	ef = create_edited_field((LatticeVol<double>*)f,
				 (LatticeVol<unsigned short>*)0);
      } else if (tn2 == "LatticeVol<int>") {
	ef = create_edited_field((LatticeVol<double>*)f,(LatticeVol<int>*)0);
      } else if (tn2 == "LatticeVol<unsigned int>") {
	ef = create_edited_field((LatticeVol<double>*)f,
				 (LatticeVol<unsigned int>*)0);
      } else if (tn2 == "LatticeVol<float>") {
	ef = create_edited_field((LatticeVol<double>*)f,(LatticeVol<float>*)0);
      } else if (tn2 == "LatticeVol<double>") {
	ef = create_edited_field((LatticeVol<double>*)f,(LatticeVol<double>*)0);
      } else if (tn2 == "LatticeVol<long>") {
	ef = create_edited_field((LatticeVol<double>*)f,(LatticeVol<long>*)0);
      }
    } else if (tn1 == "LatticeVol<long>") {
      if (tn2 == "LatticeVol<char>") {
	ef = create_edited_field((LatticeVol<long>*)f,(LatticeVol<char>*)0);
      } else if (tn2 == "LatticeVol<unsigned char>") {
	ef = create_edited_field((LatticeVol<long>*)f,
				 (LatticeVol<unsigned char>*)0);
      } else if (tn2 == "LatticeVol<short>") {
	ef = create_edited_field((LatticeVol<long>*)f,(LatticeVol<short>*)0);
      } else if (tn2 == "LatticeVol<unsigned short>") {
	ef = create_edited_field((LatticeVol<long>*)f,
				 (LatticeVol<unsigned short>*)0);
      } else if (tn2 == "LatticeVol<int>") {
	ef = create_edited_field((LatticeVol<long>*)f,(LatticeVol<int>*)0);
      } else if (tn2 == "LatticeVol<unsigned int>") {
	ef = create_edited_field((LatticeVol<long>*)f,
				 (LatticeVol<unsigned int>*)0);
      } else if (tn2 == "LatticeVol<float>") {
	ef = create_edited_field((LatticeVol<long>*)f,(LatticeVol<float>*)0);
      } else if (tn2 == "LatticeVol<double>") {
	ef = create_edited_field((LatticeVol<long>*)f,(LatticeVol<double>*)0);
      } else if (tn2 == "LatticeVol<long>") {
	ef = create_edited_field((LatticeVol<long>*)f,(LatticeVol<long>*)0);
      }
    }

  }

  oport->send(ef);
}
    
void EditField::tcl_command(TCLArgs& args, void* userdata)
{
  if(args.count() < 2){
    args.error("EditField needs a minor command");
    return;
  }
 
  if (args[1] == "execute") {
    want_to_execute();
  } else if (args[1] == "update_widget") {
    Point center, right, down, in;
    minx_.reset(); miny_.reset(); minz_.reset();
    maxx_.reset(); maxy_.reset(); maxz_.reset();
    Point min(minx_.get(),miny_.get(),minz_.get());
    Point max(maxx_.get(),maxy_.get(),maxz_.get());
    if (max.x()<=min.x() ||
	max.y()<=min.y() ||
	max.z()<=min.z()) {
      postMessage("EditField: Degenerate BBox requested!");
      return;                    // degenerate 
    }
    center = min+((max-min)/2.);
    right = Point(max.x(),center.y(),center.z());
    down = Point(center.x(),max.y(),center.z());
    in = Point(center.x(),center.y(),max.z());
    box_->SetPosition(center,right,down,in);
    want_to_execute();
  } else {
    Module::tcl_command(args, userdata);
  }
}

void EditField::widget_moved(int i)
{
  if (i==1) {
    Point center, right, down, in;
    minx_.reset(); miny_.reset(); minz_.reset();
    maxx_.reset(); maxy_.reset(); maxz_.reset();
    box_->GetPosition(center,right,down,in);
    minx_.set((center-(right-center)).x());
    miny_.set((center-(down-center)).y());
    minz_.set((center-(in-center)).z());
    maxx_.set(right.x());
    maxy_.set(down.y());
    maxz_.set(in.z());
    want_to_execute();
  } else {
    //Module::widget_moved(i);
  }
}


CompileInfo *
EditFieldAlgoCN::get_compile_info(const TypeDescription *mesh_td)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("EditFieldAlgoCNT");
  static const string base_class_name("EditFieldAlgoCN");

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       to_filename(mesh_td->get_name()) + ".",
                       base_class_name, 
                       template_class_name, 
                       mesh_td->get_name());

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  mesh_td->fill_compile_info(rval);
  return rval;
}

} // End namespace Moulding


