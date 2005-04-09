/*
 * This module makes the boundary of a scalar field integrate to 0
 * this properly sets the "reference" point for the voltage
 *
 * Peter-Pike Sloan
 */

#include <Classlib/NotFinished.h>
#include <Dataflow/Module.h>
#include <Datatypes/GeometryPort.h>
#include <Datatypes/ScalarFieldPort.h>
#include <Datatypes/ScalarFieldUG.h>
#include <Malloc/Allocator.h>
#include <iostream.h>

using sci::Mesh;

class AdjustField : public Module {
    ScalarFieldIPort* insfield;
    ScalarFieldOPort* outsfield;
public:
    AdjustField(const clString& id);
    AdjustField(const AdjustField&, int deep);
    virtual ~AdjustField();
    virtual Module* clone(int deep);
    virtual void execute();
    MaterialHandle matl;
};

extern "C" {
Module* make_AdjustField(const clString& id)
{
    return new AdjustField(id);
}
};

AdjustField::AdjustField(const clString& id)
: Module("AdjustField", id, Filter)
{
    // Create the input ports
    insfield=new ScalarFieldIPort(this, "Scalar Field", ScalarFieldIPort::Atomic);
    add_iport(insfield);
    // Create the output port
    outsfield=new ScalarFieldOPort(this, "Scalar Field", ScalarFieldIPort::Atomic);
    add_oport(outsfield);
}

AdjustField::AdjustField(const AdjustField& copy, int deep)
: Module(copy, deep)
{
    NOT_FINISHED("AdjustField::AdjustField");
}

AdjustField::~AdjustField()
{
}

Module* AdjustField::clone(int deep)
{
    return new AdjustField(*this, deep);
}

void AdjustField::execute()
{
    ScalarFieldHandle sfield;
    if(insfield->get(sfield) && sfield.get_rep() && sfield->getUG()){
      ScalarFieldUG *worker = sfield->getUG();
      ScalarFieldUG *news = scinew ScalarFieldUG(worker->mesh,worker->typ);
      Array1<int> pts;
      worker->mesh->get_boundary_nodes(pts);
      
      double avrg=0.0;
      for(int i=0;i<pts.size();i++) {
	avrg += worker->data[pts[i]];
      }
      avrg /= pts.size();

      for(i=0;i<news->data.size();i++) {
	news->data[i] = worker->data[i] - avrg;
      }

      news->compute_minmax();

      outsfield->send(news);

    }
}

