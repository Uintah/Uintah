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
  // create an example 3D field of Vectors, which is 64^3
  // and has extents equal to [-3.15,3.15]x[-3.15,3.15]x[-3.15,3.15]
  int x,y,z;
  x = y = z = 64;        // number of samples in each dimension
  double b = 3.15;   
  Point start(-b,-b,-b); // extents of the geometry.
  Point end(b,b,b);

  // create geometry and attribute objects.  Note that the
  // geometry and the attribute have the same dimensions (3D),
  // and the same number of samples (x,y,z) in each dimension.
  LatticeGeom* geom = scinew LatticeGeom(x,y,z,start,end);
  DiscreteAttrib<Vector>* attr = scinew FlatAttrib<Vector>(x,y,z);

  // create a field using the above geometry and attribute.
  GenVField<Vector,LatticeGeom>* field = 
    scinew GenVField<Vector,LatticeGeom>(geom,attr);

  // populate the field with some values.
  int i,j,k;
  double gap = b*2.0/(x-1);  // distance between nodes along the x axis

  for (i=0;i<x;i++) 
    for (j=0;j<y;j++)
      for (k=0;k<z;k++)
	attr->set3(i,j,k,Vector(1,sin(-b+i*gap),0));

  // send the field out the output port.
  FieldHandle* handle = scinew FieldHandle(field);
  oport->send(*handle);
}

void GenField::tcl_command(TCLArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace Modules
} // End namespace Moulding


