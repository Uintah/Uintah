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

namespace SCIRun {

class PSECORESHARE EditField : public Module {
public:
  GuiString name_;
  GuiString bboxmin_;
  GuiString bboxmax_;
  GuiString typename_;
  GuiString datamin_;
  GuiString datamax_;
  GuiString numnodes_;
  GuiString numelems_;
  GuiString dataat_;

  EditField(const string& id);

  virtual ~EditField();

  template <class Field>
  void set_nums(Field *);
  void clear_vals();

  virtual void execute();

  virtual void tcl_command(TCLArgs&, void*);
};

extern "C" PSECORESHARE Module* make_EditField(const string& id) {
  return scinew EditField(id);
}

EditField::EditField(const string& id)
  : Module("EditField", id, Source, "Fields", "SCIRun"),
    name_("name", id, this),
    bboxmin_("bboxmin", id, this),
    bboxmax_("bboxmax", id, this),
    typename_("typename", id, this),
    datamin_("datamin", id, this),
    datamax_("datamax", id, this),
    numnodes_("numnodes", id, this),
    numelems_("numelems", id, this),
    dataat_("dataat", id, this)
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

  numnodes_.set(to_string(count));
  count = 0;

  elem_iter ei = mesh->tbegin((elem_iter*)0);
  while (ei != mesh->tend((elem_iter*)0)) {
    count++;++ei;
  }

  numelems_.set(to_string(count));
}

void EditField::clear_vals() 
{
  name_.set("---");
  bboxmin_.set("---");
  bboxmax_.set("---");
  typename_.set("---");
  datamin_.set("---");
  datamax_.set("---");
  numnodes_.set("---");
  numelems_.set("---");
  dataat_.set("---");
  TCL::execute(id+" update_attributes");
}

void EditField::execute(){
  FieldIPort *iport = (FieldIPort*)get_iport(0);
  FieldHandle fh;
  Field *f=0;
  if (!iport || !iport->get(fh) || !(f=fh.get_rep())) {
    clear_vals();
    return;
  }

  string tname(f->get_type_name(-1));

  typename_.set(tname);
  switch(f->data_at()) {
  case Field::CELL: dataat_.set("Field::CELL"); break;
  case Field::FACE: dataat_.set("Field::FACE"); break;
  case Field::EDGE: dataat_.set("Field::EDGE"); break;
  case Field::NODE: dataat_.set("Field::NODE"); break; 
  case Field::NONE: dataat_.set("Field::NONE"); break;
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
  } else if (tname=="TirSurf<Vector>") {
    set_nums((TriSurf<Vector>*)f);
  }

  TCL::execute(id + " update_attributes");    
}

void EditField::tcl_command(TCLArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace Moulding


