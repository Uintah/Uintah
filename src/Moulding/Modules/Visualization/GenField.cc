/*
 *  GenField.cc:
 *
 *  Written by:
 *   moulding
 *   TODAY'S DATE HERE
 *
 */

#include <PSECore/Dataflow/Module.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/Datatypes/Field.h>
#include <PSECore/Datatypes/FieldPort.h>
#include <SCICore/Datatypes/VField.h>
#include <SCICore/Datatypes/GenVField.h>
#include <SCICore/Datatypes/LatticeGeom.h>
#include <SCICore/Datatypes/FlatAttrib.h>

#include <Moulding/share/share.h>

namespace Moulding {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace SCICore::Datatypes;
using namespace PSECore::Datatypes;

class MouldingSHARE GenField : public Module {
public:
  GenField(const clString& id);

  virtual ~GenField();

  virtual void execute();

  virtual void tcl_command(TCLArgs&, void*);

private:
  FieldOPort* oport;
};

extern "C" MouldingSHARE Module* make_GenField(const clString& id) {
  return new GenField(id);
}

GenField::GenField(const clString& id)
  : Module("GenField", id, Source)
{
  oport = scinew FieldOPort(this, "Sample Field", FieldIPort::Atomic);
  add_oport(oport);
}

GenField::~GenField(){
}

void GenField::execute()
{
  int n    = 16;     // number samples in each dimension
  double b = 3.15;   // extent of the samples in each dimension

  Point s(-b,-b,-b);
  Point e(b,b,b);

  LatticeGeom* geom = new LatticeGeom(n,n,n,s,e);
  DiscreteAttrib<Vector>* attr = new FlatAttrib<Vector>(n,n,n);
  GenVField<Vector,LatticeGeom>* field = 
    new GenVField<Vector,LatticeGeom>(geom,attr);

  int i,j,k;
  double gap = b*2.0/n;

  for (i=0;i<n;i++) 
    for (j=0;j<n;j++)
      for (k=0;k<n;k++)
	attr->set3(i,j,k,Vector(1,sin(-b+j*gap),0));

  FieldHandle* handle = scinew FieldHandle(field);
  oport->send(*handle);
}

void GenField::tcl_command(TCLArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace Modules
} // End namespace Moulding


