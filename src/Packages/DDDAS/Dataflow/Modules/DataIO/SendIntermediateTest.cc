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
#include <Dataflow/Network/Ports/FieldPort.h>
#include <Core/Basis/Constant.h>
#include <Core/Datatypes/PointCloudMesh.h>
#include <Core/Datatypes/GenericField.h>

namespace DDDAS {

using namespace SCIRun;
typedef ConstantBasis<double>                               PCDatBasis;
typedef PointCloudMesh<ConstantBasis<Point> >               PCMesh;
typedef GenericField<PCMesh, PCDatBasis, vector<double> >   PCField;

class SendIntermediateTest : public Module {
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
  PCMesh *pcm = scinew PCMesh();
  Point p(0,0,0);
  pcm->add_point(p);
  PCField *pcfd =  scinew PCField(pcm);
  pcfd->resize_fdata();
  pcfd->set_value(69.69, (PCMesh::Node::index_type)0);
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


