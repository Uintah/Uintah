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

#include <Dataflow/Ports/FieldPort.h>
#include <Core/Datatypes/LatticeVol.h>
#include <Core/Datatypes/TetVol.h>
#include <Core/Datatypes/TriSurf.h>

#include <map>

namespace SCIRun {

using std::pair;

class PSECORESHARE EditField : public Module {
public:
  GuiString fldname_;
  GuiString typename_;
  GuiString datamin_;
  GuiString datamax_;
  GuiString numnodes_;
  GuiString numelems_;
  GuiString dataat_;
  GuiDouble minx_,miny_,minz_;
  GuiDouble maxx_,maxy_,maxz_;

  GuiInt cfldname_;
  GuiInt ctypename_;
  GuiInt cnumelems_;
  GuiInt cdataat_;
  GuiInt cdataminmax_;

  EditField(const string& id);

  virtual ~EditField();

  template <class Field>
  void set_nums(Field *);
  void clear_vals();
  void update_input_attributes(Field *);

  virtual void execute();

  virtual void tcl_command(TCLArgs&, void*);
};

extern "C" PSECORESHARE Module* make_EditField(const string& id) {
  return scinew EditField(id);
}

EditField::EditField(const string& id)
  : Module("EditField", id, Source, "Fields", "SCIRun"),
    fldname_("fldname2", id, this),
    typename_("typename2", id, this),
    datamin_("datamin2", id, this),
    datamax_("datamax2", id, this),
    numnodes_("numnodes2", id, this),
    numelems_("numelems2", id, this),
    dataat_("dataat2", id, this),
    minx_("minx2", id, this),
    miny_("miny2", id, this),
    minz_("minz2", id, this),
    maxx_("maxx2", id, this),
    maxy_("maxy2", id, this),
    maxz_("maxz2", id, this),
    cfldname_("cfldname", id, this),
    ctypename_("ctypename", id, this),
    cnumelems_("cnumelems", id, this),
    cdataat_("cdataat", id, this),
    cdataminmax_("cdataminmax", id, this)
{
}

EditField::~EditField(){
}

double mag_val(double v) { return v; }
double mag_val(Vector v) { return v.length(); }

template <class Field>
void 
EditField::set_nums(Field *f) {
  typedef typename Field::mesh_type          mesh_type;
  typedef typename mesh_type::Node::iterator node_iter;
  typedef typename mesh_type::Elem::iterator elem_iter;

  int count = 0;
  mesh_type *mesh = f->get_typed_mesh().get_rep();

  node_iter ni = mesh->tbegin((node_iter*)0);
  while (ni != mesh->tend((node_iter*)0)) {
    count++;++ni;
  }

  TCL::execute(string("set ")+id+"-numnodes "+to_string(count));
  count = 0;

  elem_iter ei = mesh->tbegin((elem_iter*)0);
  while (ei != mesh->tend((elem_iter*)0)) {
    count++;++ei;
  }

  TCL::execute(string("set ")+id+"-numelems "+to_string(count));
}

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

void EditField::update_input_attributes(Field *f) 
{
  string tname(f->get_type_name(-1));

  TCL::execute(string("set ")+id+"-typename "+tname);

  switch(f->data_at()) {
  case Field::CELL: 
    TCL::execute(string("set ")+id+"-dataat Field::CELL"); break;
  case Field::FACE: 
    TCL::execute(string("set ")+id+"-dataat Field::FACE"); break;
  case Field::EDGE: 
    TCL::execute(string("set ")+id+"-dataat Field::EDGE"); break;
  case Field::NODE: 
    TCL::execute(string("set ")+id+"-dataat Field::NODE"); break;
  case Field::NONE: 
    TCL::execute(string("set ")+id+"-dataat Field::NONE"); break;
  }

  if (tname=="TetVol<double>") {
    set_nums((TetVol<double>*)f);
  } else if (tname=="TetVol<Vector>") {
    set_nums((TetVol<Vector>*)f);
  } else if (tname=="LatticeVol<double>") {
    set_nums((LatticeVol<double>*)f);
  } else if (tname=="LatticeVol<Vector>") {
    set_nums((LatticeVol<Vector>*)f);
  } else if (tname=="TriSurf<double>") {
    set_nums((TriSurf<double>*)f);
  } else if (tname=="TriSurf<Vector>") {
    set_nums((TriSurf<Vector>*)f);
  }

  const BBox bbox = f->mesh()->get_bounding_box();
  Point min = bbox.min();
  Point max = bbox.max();

  TCL::execute(string("set ")+id+"-minx "+to_string(min.x()));
  TCL::execute(string("set ")+id+"-miny "+to_string(min.y()));
  TCL::execute(string("set ")+id+"-minz "+to_string(min.z()));
  TCL::execute(string("set ")+id+"-maxx "+to_string(max.x()));
  TCL::execute(string("set ")+id+"-maxy "+to_string(max.y()));
  TCL::execute(string("set ")+id+"-maxz "+to_string(max.z()));

  pair<double,double> minmax(1,0);
  if (f->get("minmax",minmax)) {
    TCL::execute(string("set ")+id+"-datamin "+to_string(minmax.first));
    TCL::execute(string("set ")+id+"-datamax "+to_string(minmax.second));
  } else {
    TCL::execute(string("set ")+id+"-datamin \"--- Not Applicable ---\"");
    TCL::execute(string("set ")+id+"-datamax \"--- Not Applicable ---\"");
  }
  
  string fldname;
  if (f->get("name",fldname))
    TCL::execute(string("set ")+id+"-fldname "+fldname);
  else
    TCL::execute(string("set ")+id+"-fldname \"--- Name Not Assigned ---\"");

  TCL::execute(id+" update_multifields");
}

void EditField::execute(){
  FieldIPort *iport=0; 
  FieldHandle fh;
  Field *f=0;
  if (!(iport=(FieldIPort*)get_iport(0)) || 
      !iport->get(fh) || 
      !(f=fh.get_rep())) {
    clear_vals();
    return;
  }

  update_input_attributes(f);
  TCL::execute(id+" update_multifields");

  GuiString fldname("fldname2", id, this);
  GuiString type_name("typename2", id, this);
  GuiString datamin("datamin2", id, this);
  GuiString datamax("datamax2", id, this);
  GuiString numelems("numelems2", id, this);
  GuiString dataat("dataat2", id, this);
  GuiDouble minx("minx2",id,this),miny("miny2",id,this),minz("minz2",id,this);
  GuiDouble maxx("maxx2",id,this),maxy("maxy2",id,this),maxz("maxz2",id,this);

  GuiInt cfldname("cfldname", id, this);
  GuiInt ctypename("ctypename", id, this);
  GuiInt cnumelems("cnumelems", id, this);
  GuiInt cdataat("cdataat", id, this);
  GuiInt cdataminmax("cdataminmax", id, this);

  std::cerr << "fldname : " << fldname.get() << " " << cfldname.get() 
	    << std::endl; 
}
    
void EditField::tcl_command(TCLArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace Moulding


