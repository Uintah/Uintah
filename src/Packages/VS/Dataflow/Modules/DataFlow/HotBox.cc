/*****************************************************************************
 * File: SCIRun/src/Packages/VS/Dataflow/Modules/DataFlow/HotBox.cc
 *
 * Description: C++ source implementation of the Virtual Soldier HotBox
 *              UI Module in SCIRun.
 *
 * Written by:
 *   Stewart Dickson <mailto:dicksonsp@ornl.gov>
 *                   <http://www.csm.ornl.gov/~dickson>
 *
 *   Thu Mar 11 10:19:45 EST 2004
 *
 *****************************************************************************/

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Core/Datatypes/FieldInterface.h>
#include <Core/Datatypes/PointCloudField.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Geom/GeomGroup.h>
#include <Core/Geom/GeomText.h>
#include <Core/Geom/GeomLine.h>
#include <Core/Geom/ColorMapTex.h>
#include <Core/Geom/GeomTransform.h>
#include <Core/Geometry/Transform.h>
#include <Core/Geom/GeomSticky.h>

#include <Dataflow/share/share.h>

#include <sys/stat.h>
#include <string.h>
#include <iostream>
#include "VS_SCI_HotBox.h"
#include "labelmaps.h"

namespace VS {

using namespace SCIRun;


class PSECORESHARE HotBox : public Module {
private:

  // the data from the input MatrixPort (Probe Element Index)
  struct MatrixData {
	int nrows, ncols;
  };

  // values from the Knowledgebase Query -> UI
  // GuiString gui_entity_name_[9];
  GuiString gui_label1_;
  GuiString gui_label2_;
  GuiString gui_label3_;
  GuiString gui_label4_;
  GuiString gui_label5_;
  GuiString gui_label6_;
  GuiString gui_label7_;
  GuiString gui_label8_;
  GuiString gui_label9_;

  // the HotBox interaction
  VS_SCI_Hotbox *VS_HotBoxUI;

  // temporary:  fixed anatomical label map files
  GuiString anatomydatasource_;
  GuiString adjacencydatasource_;
  VH_MasterAnatomy *anatomytable;
  VH_AdjacencyMapping *adjacencytable;

public:
  HotBox(GuiContext*);

  virtual ~HotBox();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);
};

/*
 * Default Constructor
 */

DECLARE_MAKER(HotBox)

HotBox::HotBox(GuiContext* ctx)
  : Module("HotBox", ctx, Filter, "DataFlow", "VS"),
  gui_label1_(ctx->subVar("gui_label1")),
  gui_label2_(ctx->subVar("gui_label2")),
  gui_label3_(ctx->subVar("gui_label3")),
  gui_label4_(ctx->subVar("gui_label4")),
  gui_label5_(ctx->subVar("gui_label5")),
  gui_label6_(ctx->subVar("gui_label6")),
  gui_label7_(ctx->subVar("gui_label7")),
  gui_label8_(ctx->subVar("gui_label8")),
  gui_label9_(ctx->subVar("gui_label9")),
  anatomydatasource_(ctx->subVar("anatomydatasource")),
  adjacencydatasource_(ctx->subVar("adjacencydatasource"))
{
  // instantiate the HotBox-specific interaction structure
  VS_HotBoxUI = new VS_SCI_Hotbox();
  // Start the Java Virtual Machine
  // Initialize our interface (JNI via JACE)
  // to the Protege Foundational Model of Anatomy (FMA)
  // temporary -- use fixed text files
  anatomytable = new VH_MasterAnatomy();
  adjacencytable = new VH_AdjacencyMapping();
}

HotBox::~HotBox(){
  delete anatomytable;
  delete adjacencytable;
}

void
 HotBox::execute(){

  // get input field port
  FieldIPort *inputFieldPort = (FieldIPort *)get_iport("Input Field");
  if (!inputFieldPort) {
    error("Unable to initialize input field port.");
    return;
  }
 
  // get handle to field from the input port
  FieldHandle InputFieldHandle;
  if (!inputFieldPort->get(InputFieldHandle) ||
      !InputFieldHandle.get_rep())
  {
    remark("No data on input field port.");
    return;
  }

  int labelIndexVal;

  if(InputFieldHandle->query_scalar_interface(this).get_rep() != 0)
  {
    // we have scalar data in the input field
    remark("Scalar Data on Input");

    // assume we have a PointCloudField on input containing a single point
    PointCloudField<double> *
    dpcf = dynamic_cast<PointCloudField<double>*>
	(InputFieldHandle.get_rep());
    if (dpcf != 0)
    {
      // we expect only one node in this field,
      // get the field value at this node.
      // It should be the LabelMap Index that HotBox needs.
      float inputVal = dpcf->value((PointCloudMesh::Node::index_type)0); 
      labelIndexVal = (int)inputVal;
      cout << "VS/Hotbox::Label value<double>: " << labelIndexVal << endl;
    }
    else
    {
       remark("Input Field is not of type PointCloudField<unsigned short>.");
    }

  }
  else if(InputFieldHandle->query_vector_interface(this).get_rep() != 0)
  {
    // we have vector data in the input field
    remark("Vector Data on Input");
  }
  else if(InputFieldHandle->query_tensor_interface(this).get_rep() != 0)
  {
    // we have tensor data in the input field
    remark("Tensor Data on Input");
  }

  // get input matrix port
  MatrixIPort *inputMatrixPort = (MatrixIPort *)get_iport("Input Matrix");
  if (!inputMatrixPort) {
    error("Unable to initialize input matrix port.");
    return;
  }
                                                                                
  // get handle to matrix from the input port
  MatrixHandle inputMatrixHandle;
  if(!inputMatrixPort->get(inputMatrixHandle) ||
     !inputMatrixHandle.get_rep())
  {
    remark("No data on input matrix port.");
    return;
  }

  // get the matrix data
  //  Matrix *matrixPtr = inputMatrixHandle.get_rep();

  const string anatomyDataSrc(anatomydatasource_.get());
  const string adjacencyDataSrc(adjacencydatasource_.get());
  
  // // launch queries via JACE/Protege
  //
  // // collect query results

  // for now, use fixed labelMap files
  // Read the status of this file so we can compare modification timestamps
  struct stat buf;
  if( anatomyDataSrc == "" ) {
    error("No MasterAnatomy file has been selected.  Please choose a file.");
    return;
  } else if (stat(anatomyDataSrc.c_str(), &buf)) {
    error("File '" + anatomyDataSrc + "' not found.");
    return;
  }
  if(!anatomytable->get_num_names())
  { // label maps have not been read
    anatomytable->readFile((char *)anatomyDataSrc.c_str());
  }
  else
  {
    cout << "Master Anatomy file contains " << anatomytable->get_num_names();
    cout << " names" << endl;
  }

  if(!adjacencytable->get_num_names())
  { // adjacency data has not been read
    adjacencytable->readFile((char *)adjacencyDataSrc.c_str());
  }
  else
  {
    cout << "Adjacency Map file contains " << adjacencytable->get_num_names();
    cout << " entries" << endl;
  }

  char *selectName = anatomytable->get_anatomyname(labelIndexVal);
  if(selectName != 0)
    cout << "VS/HotBox: selected '" << selectName << "'" << endl;
  else
    remark("Selected [NULL]");

  // draw HotBox Widget
  GeomGroup *HB_geomGroup = scinew GeomGroup();
  Color text_color;
  text_color = Color(1,1,1);
  MaterialHandle text_material = scinew Material(text_color);

  GeomLines *lines = scinew GeomLines();
  GeomTexts *texts = scinew GeomTexts();

  VS_HotBoxUI->setOutput(lines, texts);
  VS_HotBoxUI->setOutMtl(text_material);

  // get the adjacency info for the selected entity
  int *adjPtr = adjacencytable->adjacent_to(labelIndexVal);

  // fill in text labels in the HotBox
  char *adjacentName;
  if(adjacencytable->get_num_rel(labelIndexVal) >= 1)
  {
    if(adjPtr[1] < anatomytable->get_num_names())
        adjacentName = anatomytable->get_anatomyname(adjPtr[1]);
    else
    {
      cerr << "HotBox::execute(): adjacent index[" << adjPtr[1];
      cerr << "] out of range" << endl;
      // set result to "unknown"
      adjacentName = anatomytable->get_anatomyname(0);
    }
    cerr << "HotBox::execute(): adjacent[" << adjPtr[1] << "]: ";
    cerr << adjacentName << endl;
    gui_label1_.set(adjacentName);
    VS_HotBoxUI->set_text(1, strdup(adjacentName));
  }
  if(adjacencytable->get_num_rel(labelIndexVal) >= 2)
  {
    if(adjPtr[2] < anatomytable->get_num_names())
        adjacentName = anatomytable->get_anatomyname(adjPtr[2]);
    else
    {
      cerr << "HotBox::execute(): adjacent index[" << adjPtr[2];
      cerr << "] out of range" << endl;
      // set result to "unknown"
      adjacentName = anatomytable->get_anatomyname(0);
    }
    cerr << "HotBox::execute(): adjacent[" << adjPtr[2] << "]: ";
    cerr << adjacentName << endl;
    gui_label2_.set(adjacentName);
    VS_HotBoxUI->set_text(2, strdup(adjacentName));
  }
  if(adjacencytable->get_num_rel(labelIndexVal) >= 3)
  {
    if(adjPtr[3] < anatomytable->get_num_names())
        adjacentName = anatomytable->get_anatomyname(adjPtr[3]);
    else
    {
      cerr << "HotBox::execute(): adjacent index[" << adjPtr[3];
      cerr << "] out of range" << endl;
      // set result to "unknown"
      adjacentName = anatomytable->get_anatomyname(0);
    }
    cerr << "HotBox::execute(): adjacent[" << adjPtr[3] << "]: ";
    cerr << adjacentName << endl;
    gui_label3_.set(adjacentName);
    VS_HotBoxUI->set_text(3, strdup(adjacentName));
  }
  if(adjacencytable->get_num_rel(labelIndexVal) >= 4)
  {
    if(adjPtr[4] < anatomytable->get_num_names())
        adjacentName = anatomytable->get_anatomyname(adjPtr[4]);
    else
    {
      cerr << "HotBox::execute(): adjacent index[" << adjPtr[4];
      cerr << "] out of range" << endl;
      // set result to "unknown"
      adjacentName = anatomytable->get_anatomyname(0);
    }
    cerr << "HotBox::execute(): adjacent[" << adjPtr[4] << "]: ";
    cerr << adjacentName << endl;
    gui_label4_.set(adjacentName);
    VS_HotBoxUI->set_text(4, strdup(adjacentName));
  }

  gui_label5_.set(selectName);
  VS_HotBoxUI->set_text(5, strdup(selectName));
  
  if(adjacencytable->get_num_rel(labelIndexVal) >= 6)
  {
    if(adjPtr[6] < anatomytable->get_num_names())
        adjacentName = anatomytable->get_anatomyname(adjPtr[6]);
    else
    {
      cerr << "HotBox::execute(): adjacent index[" << adjPtr[6];
      cerr << "] out of range" << endl;
      // set result to "unknown"
      adjacentName = anatomytable->get_anatomyname(0);
    }
    cerr << "HotBox::execute(): adjacent[" << adjPtr[6] << "]: ";
    cerr << adjacentName << endl;
    gui_label6_.set(adjacentName);
    VS_HotBoxUI->set_text(6, strdup(adjacentName));
  }
  if(adjacencytable->get_num_rel(labelIndexVal) >= 7)
  {
    if(adjPtr[7] < anatomytable->get_num_names())
        adjacentName = anatomytable->get_anatomyname(adjPtr[7]);
    else
    {
      cerr << "HotBox::execute(): adjacent index[" << adjPtr[7];
      cerr << "] out of range" << endl;
      // set result to "unknown"
      adjacentName = anatomytable->get_anatomyname(0);
    }
    cerr << "HotBox::execute(): adjacent[" << adjPtr[7] << "]: ";
    cerr << adjacentName << endl;
    gui_label7_.set(adjacentName);
    VS_HotBoxUI->set_text(7, strdup(adjacentName));
  }
  if(adjacencytable->get_num_rel(labelIndexVal) >= 8)
  {
    if(adjPtr[8] < anatomytable->get_num_names())
        adjacentName = anatomytable->get_anatomyname(adjPtr[8]);
    else
    {
      cerr << "HotBox::execute(): adjacent index[" << adjPtr[8];
      cerr << "] out of range" << endl;
      // set result to "unknown"
      adjacentName = anatomytable->get_anatomyname(0);
    }
    cerr << "HotBox::execute(): adjacent[" << adjPtr[8] << "]: ";
    cerr << adjacentName << endl;
    gui_label8_.set(adjacentName);
    VS_HotBoxUI->set_text(8, strdup(adjacentName));
  }
  if(adjacencytable->get_num_rel(labelIndexVal) >= 9)
  {
    if(adjPtr[9] < anatomytable->get_num_names())
        adjacentName = anatomytable->get_anatomyname(adjPtr[9]);
    else
    {
      cerr << "HotBox::execute(): adjacent index[" << adjPtr[9];
      cerr << "] out of range" << endl;
      // set result to "unknown"
      adjacentName = anatomytable->get_anatomyname(0);
    }
    cerr << "HotBox::execute(): adjacent[" << adjPtr[9] << "]: ";
    cerr << adjacentName << endl;
    gui_label9_.set(adjacentName);
    VS_HotBoxUI->set_text(9, strdup(adjacentName));
  }

  VS_HotBoxUI->draw(0, 0, 0.005);
  
  HB_geomGroup->add(lines);
  HB_geomGroup->add(texts);

  // set output
  GeometryOPort *outGeomPort = (GeometryOPort *)get_oport("HotBox Widget");
  if(!outGeomPort) {
    error("Unable to initialize output geometry port.");
    return;
  }
  GeomSticky *sticky = scinew GeomSticky(HB_geomGroup);
  outGeomPort->delAll();
  outGeomPort->addObj( sticky, "HotBox Sticky" );
  outGeomPort->flushViews();

} // end HotBox::execute()

void
 HotBox::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace VS


