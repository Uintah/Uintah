/*
 *  SendIntermediateTest.cc:
 *
 *  Written by:
 *   mjc
 *   TODAY'S DATE HERE
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Datatypes/PointCloudField.h>
#include <Dataflow/share/share.h>

namespace DDDAS {

using namespace SCIRun;

class PSECORESHARE SendIntermediateTest : public Module {
public:
  SendIntermediateTest(GuiContext*);

  virtual ~SendIntermediateTest();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);
  FieldHandle output_;
};


DECLARE_MAKER(SendIntermediateTest)
SendIntermediateTest::SendIntermediateTest(GuiContext* ctx) : 
  Module("SendIntermediateTest", ctx, Source, "DataIO", "DDDAS"),
  output_(0)
{
  PointCloudMesh *pcm = scinew PointCloudMesh();
  Point p(0,0,0);
  pcm->add_point(p);
  PointCloudField<double> *pcfd = 
    scinew PointCloudField<double>(pcm, 0);
  pcfd->resize_fdata();
  pcfd->set_value(69.69, (PointCloudMesh::Node::index_type)0);
  output_ = pcfd;
}

SendIntermediateTest::~SendIntermediateTest()
{
  output_ = 0;
}

void
SendIntermediateTest::execute()
{
  FieldOPort *ofp = (FieldOPort *)get_oport("continuous output");
  while(true) {
    ofp->send_intermediate(output_);
  }

}

void
 SendIntermediateTest::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace DDDAS


