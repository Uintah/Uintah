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

#include <Core/Containers/HashTable.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/CurveField.h>
#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/FieldInterface.h>
#include <Core/Datatypes/QuadSurfField.h>
#include <Core/Datatypes/QuadSurfMesh.h>
#include <Core/Datatypes/TriSurfField.h>
#include <Core/Datatypes/LatVolField.h>
#include <Core/Geom/GeomGroup.h>
#include <Core/Geom/GeomText.h>
#include <Core/Geom/GeomLine.h>
#include <Core/Geom/ColorMap.h>
#include <Core/Geom/ColorMapTex.h>
#include <Core/Geom/GeomTransform.h>
#include <Core/Geometry/Transform.h>
#include <Core/Geom/GeomSticky.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Persistent/Pstreams.h>

#include <Dataflow/Modules/Fields/Probe.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/ColorMapPort.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Dataflow/Ports/NrrdPort.h>
#include <Dataflow/Ports/TimePort.h>
#include <Dataflow/Widgets/PointWidget.h>
#include <Dataflow/XMLUtil/XMLUtil.h>
#include <Dataflow/XMLUtil/StrX.h>

#include <sys/stat.h>
#include <math.h>
#include <string.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <sgi_stl_warnings_on.h>
// Foundational Model of Anatomy Web Services
#include "soapServiceInterfaceSoapBindingProxy.h" // get proxy
#include "ServiceInterfaceSoapBinding.nsmap" // get namespace bindings
#include "stdsoap2.h"
// Xerces XML parser
#include <xercesc/framework/MemBufInputSource.hpp>
#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/dom/DOMNamedNodeMap.hpp>
#include <xercesc/dom/DOMNodeList.hpp>
#include <xercesc/dom/DOMNamedNodeMap.hpp>
#include <xercesc/sax/ErrorHandler.hpp>
#include <xercesc/sax/SAXException.hpp>
#include <xercesc/sax/SAXParseException.hpp>
// VS/Hotbox
#include "VS_SCI_HotBox.h"
#include "labelmaps.h"

namespace VS {

using namespace SCIRun;

// Query Data Source
#define VS_DATASOURCE_OQAFMA 1
#define VS_DATASOURCE_FILES 2

// Query Type
#define VS_QUERYTYPE_ADJACENT_TO 1
#define VS_QUERYTYPE_CONTAINS 2
#define VS_QUERYTYPE_PARTS 3
#define VS_QUERYTYPE_PARTCONTAINS 4

#define CYLREZ 20
#define MIN3(x, y, z) MIN(x, MIN(y, z))

class TimeSync;

class HotBox : public Module {
private:

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
  // tags
  GuiInt gui_is_injured1_;
  GuiInt gui_is_injured2_;
  GuiInt gui_is_injured3_;
  GuiInt gui_is_injured4_;
  GuiInt gui_is_injured5_;
  GuiInt gui_is_injured6_;
  GuiInt gui_is_injured7_;
  GuiInt gui_is_injured8_;
  GuiInt gui_is_injured9_;
  // diagnosis
  GuiString gui_diagnosis_;
  // hierarchical relations -- 0-based in GUI ListBox
  // this stores the Tcl name of the HotBox GUI instance
  GuiString gui_name_;
  // these store the array values
  GuiString gui_parent0_;
  GuiString gui_parent1_;
  GuiString gui_parent2_;
  GuiString gui_parent3_;
  GuiString gui_parent4_;
  GuiString gui_parent5_;
  GuiString gui_parent6_;
  GuiString gui_parent7_;
  // this stores the name of the list
  GuiString gui_parent_list_;
  // these store the array values
  GuiString gui_sibling0_;
  GuiString gui_sibling1_;
  GuiString gui_sibling2_;
  GuiString gui_sibling3_;
  // this stores the name of the list
  GuiString gui_sibling_list_;
  // these store the array values
  GuiString gui_child0_;
  GuiString gui_child1_;
  GuiString gui_child2_;
  GuiString gui_child3_;
  GuiString gui_child4_;
  GuiString gui_child5_;
  GuiString gui_child6_;
  GuiString gui_child7_;
  GuiString gui_child8_;
  GuiString gui_child9_;
  GuiString gui_child10_;
  GuiString gui_child11_;
  GuiString gui_child12_;
  GuiString gui_child13_;
  GuiString gui_child14_;
  GuiString gui_child15_;
  // this stores the name of the list
  GuiString gui_child_list_;

  // toggle on/off drawing GeomSticky output
  GuiString enableDraw_;

  // the HotBox/Viewer interaction
  VS_SCI_Hotbox *VS_HotBoxUI_;

  // file or OQAFMA
  GuiInt datasource_;

  // Query Type
  GuiInt querytype_;

  // "fromHotBoxUI" or "fromProbe"
  GuiString selectionsource_;

  // data sources -- conceptually 'URIs'
  GuiString anatomydatasource_;
  GuiString adjacencydatasource_;
  GuiString boundingboxdatasource_;
  GuiString injurylistdatasource_;
  GuiString oqafmadatasource_;
  GuiString geometrypath_;
  GuiString hipvarpath_;

  // the current selection
  GuiString currentselection_;

  // current time
  GuiString gui_curTime_;
  double currentTime_, lastTime_, timeEps_;

  // a thread to update time from input port
  TimeSync *timeSyncer_;
  Thread *timeSyncer_thread_;

  // fixed anatomical label map files
  VH_MasterAnatomy *anatomytable_;
  VH_AdjacencyMapping *adjacencytable_;
  VH_AnatomyBoundingBox *boundBoxList_;
  VH_AnatomyBoundingBox *maxSegmentVol_;
  // transform bounding boxes into field volume
  Point boxTran_, boxScale_;

  // physiology parameter mapping
  VH_HIPvarMap *hipVarFileList_;

  // the injured tissue list
  XercesDOMParser injListParser_;
  DOMDocument *injListDoc_;
  // a list of injured tissues for each timeStep
  vector <vector <VH_injury> *> injured_tissue_list_;
  vector <VH_injury> *injured_tissue_;

  // the probe widget
  PointWidget *probeWidget_;
  CrowdMonitor probeWidget_lock_;
  GuiDouble gui_probeLocx_;
  GuiDouble gui_probeLocy_;
  GuiDouble gui_probeLocz_;
  GuiDouble gui_probe_scale_;

  FieldHandle InputFieldHandle_;
  TimeViewerHandle time_viewer_handle_;
  NrrdDataHandle InputNrrdHandle_;
  int probeWidgetid_;
  Point probeLoc_;
  int labelIndexVal_;
  BBox inFieldBBox_;
  Point maxSegBBextent_;
  double l2norm_;

  // private methods
  void executeProbe();
  void execAdjacency();
  void execSelColorMap();
  void traverseDOMtree(DOMNode &, int, double *, VH_injury **);
  int get_timeStep(double);
  void parseInjuryList();
  void execInjuryList();
  void makeInjGeometry();
  void executeOQAFMA();
  void executeHighlight();
  void executePhysio();

protected:
  // output injury icon geometry (sphere, cone, ...)
  FieldHandle injIconFieldHandle_;
  Transform    inputTransform_;
  // input selected surface geometry -- from field file
  FieldHandle  selectGeomFilehandle_;
  // input injured surface geometry -- from field files
  FieldHandle  inj0GeomFilehandle_;
  FieldHandle  inj1GeomFilehandle_;
  // output injured surface geometry -- created from input mesh + data
  FieldHandle  inj0GeomFieldhandle_;
  FieldHandle  inj1GeomFieldhandle_;
  string    geomFilename_;
  string    activeBoundBoxSrc_;
  string    activeInjList_;
  string    lastSelection_;
  string    currentSelection_;

public:
  HotBox(GuiContext*);

  virtual ~HotBox();

  virtual void execute();
  virtual void widget_moved(bool, BaseWidget*);

  virtual void tcl_command(GuiArgs&, void*);

  void sync_time(double t);

}; // end class HotBox

class TimeSync : public Runnable {
public:
  TimeSync(HotBox *module, TimeViewerHandle tvh) :
    module_(module),
    throttle_(),
    tvh_(tvh),
    dead_(0)
  {};
  TimeSync();
  ~TimeSync();
  virtual void run();
  void set_dead(bool p) { dead_ = p; }
private:
  HotBox *module_;
  TimeThrottle           throttle_;
  TimeViewerHandle       tvh_;
  bool                   dead_;
};

TimeSync::~TimeSync()
{
}

void
TimeSync::run()
{
  throttle_.start();
  const double inc = 1./5.; // the rate at which we refresh the time stamp
  double t = throttle_.time();
  while (!dead_)
  {
    t = throttle_.time();
    throttle_.wait_for_time(t + inc);
    module_->sync_time(tvh_->view_elapsed_since_start());
  }
} // end TimeSync::run()

/*****************************************************************************
 * Default Constructor
 *****************************************************************************/

DECLARE_MAKER(HotBox)

HotBox::HotBox(GuiContext* ctx)
  : Module("HotBox", ctx, Filter, "DataFlow", "VS"),
  gui_label1_(ctx->subVar("gui_label(1)")),
  gui_label2_(ctx->subVar("gui_label(2)")),
  gui_label3_(ctx->subVar("gui_label(3)")),
  gui_label4_(ctx->subVar("gui_label(4)")),
  gui_label5_(ctx->subVar("gui_label(5)")),
  gui_label6_(ctx->subVar("gui_label(6)")),
  gui_label7_(ctx->subVar("gui_label(7)")),
  gui_label8_(ctx->subVar("gui_label(8)")),
  gui_label9_(ctx->subVar("gui_label(9)")),
  gui_is_injured1_(ctx->subVar("gui_is_injured(1)")),
  gui_is_injured2_(ctx->subVar("gui_is_injured(2)")),
  gui_is_injured3_(ctx->subVar("gui_is_injured(3)")),
  gui_is_injured4_(ctx->subVar("gui_is_injured(4)")),
  gui_is_injured5_(ctx->subVar("gui_is_injured(5)")),
  gui_is_injured6_(ctx->subVar("gui_is_injured(6)")),
  gui_is_injured7_(ctx->subVar("gui_is_injured(7)")),
  gui_is_injured8_(ctx->subVar("gui_is_injured(8)")),
  gui_is_injured9_(ctx->subVar("gui_is_injured(9)")),
  gui_diagnosis_(ctx->subVar("gui_diagnosis")),
  gui_name_(ctx->subVar("gui_name")),
  gui_parent0_(ctx->subVar("gui_parent(0)")),
  gui_parent1_(ctx->subVar("gui_parent(1)")),
  gui_parent2_(ctx->subVar("gui_parent(2)")),
  gui_parent3_(ctx->subVar("gui_parent(3)")),
  gui_parent4_(ctx->subVar("gui_parent(4)")),
  gui_parent5_(ctx->subVar("gui_parent(5)")),
  gui_parent6_(ctx->subVar("gui_parent(6)")),
  gui_parent7_(ctx->subVar("gui_parent(7)")),
  gui_parent_list_(ctx->subVar("gui_parlist_name")),
  gui_sibling0_(ctx->subVar("gui_sibling(0)")),
  gui_sibling1_(ctx->subVar("gui_sibling(1)")),
  gui_sibling2_(ctx->subVar("gui_sibling(2)")),
  gui_sibling3_(ctx->subVar("gui_sibling(3)")),
  gui_sibling_list_(ctx->subVar("gui_siblist_name")),
  gui_child0_(ctx->subVar("gui_child(0)")),
  gui_child1_(ctx->subVar("gui_child(1)")),
  gui_child2_(ctx->subVar("gui_child(2)")),
  gui_child3_(ctx->subVar("gui_child(3)")),
  gui_child4_(ctx->subVar("gui_child(4)")),
  gui_child5_(ctx->subVar("gui_child(5)")),
  gui_child6_(ctx->subVar("gui_child(6)")),
  gui_child7_(ctx->subVar("gui_child(7)")),
  gui_child8_(ctx->subVar("gui_child(8)")),
  gui_child9_(ctx->subVar("gui_child(9)")),
  gui_child10_(ctx->subVar("gui_child(10)")),
  gui_child11_(ctx->subVar("gui_child(11)")),
  gui_child12_(ctx->subVar("gui_child(12)")),
  gui_child13_(ctx->subVar("gui_child(13)")),
  gui_child14_(ctx->subVar("gui_child(14)")),
  gui_child15_(ctx->subVar("gui_child(15)")),
  gui_child_list_(ctx->subVar("gui_childlist_name")),
  enableDraw_(ctx->subVar("enableDraw")),
  datasource_(ctx->subVar("datasource")),
  querytype_(ctx->subVar("querytype")),
  selectionsource_(ctx->subVar("selectionsource")),
  anatomydatasource_(ctx->subVar("anatomydatasource")),
  adjacencydatasource_(ctx->subVar("adjacencydatasource")),
  boundingboxdatasource_(ctx->subVar("boundingboxdatasource")),
  injurylistdatasource_(ctx->subVar("injurylistdatasource")),
  oqafmadatasource_(ctx->subVar("oqafmadatasource")),
  geometrypath_(ctx->subVar("geometrypath")),
  hipvarpath_(ctx->subVar("hipvarpath")),
  currentselection_(ctx->subVar("currentselection")),
  gui_curTime_(ctx->subVar("currentTime")),
  currentTime_(0.0),
  lastTime_(-1000.0),
  timeEps_(0.25),
  timeSyncer_(0),
  timeSyncer_thread_(0),
  injListDoc_(0),
  probeWidget_lock_("PointWidget lock"),
  gui_probeLocx_(ctx->subVar("gui_probeLocx")),
  gui_probeLocy_(ctx->subVar("gui_probeLocy")),
  gui_probeLocz_(ctx->subVar("gui_probeLocz")),
  gui_probe_scale_(ctx->subVar("gui_probe_scale")),
  probeWidgetid_(-1)
{
  // instantiate the HotBox-specific interaction structure
  VS_HotBoxUI_ = new VS_SCI_Hotbox();
  // instantiate label maps
  anatomytable_ = new VH_MasterAnatomy();
  adjacencytable_ = new VH_AdjacencyMapping();
  boundBoxList_ = (VH_AnatomyBoundingBox *)NULL;
  hipVarFileList_ = new VH_HIPvarMap();
  // create the probe widget
  probeWidget_ = scinew PointWidget(this, &probeWidget_lock_, 1.0);
  probeWidget_->Connect((GeometryOPort*)get_oport("Probe Widget"));
} // end HotBox::HotBox()

HotBox::~HotBox()
{
  delete anatomytable_;
  delete adjacencytable_;
  delete probeWidget_;
  delete hipVarFileList_;
  if(timeSyncer_thread_)
  {
    timeSyncer_->set_dead(true);
    timeSyncer_thread_->join();
    timeSyncer_thread_ = 0;
  }
} // end HotBox::~HotBox()

void
HotBox::execute()
{

  // get input Time port
  TimeIPort *
  inputTimePort = (TimeIPort *)get_iport("Time");
  if (!inputTimePort)
  {
    error("Unable to initialize input Time port.");
  }
  else
  {
    inputTimePort->get(time_viewer_handle_);
  }

  // get a handle to the time viewer from the input Time port
  if(inputTimePort != NULL &&
     !time_viewer_handle_.get_rep())
  {
    remark("No data on input Time port.");
  }
  else
  { // get/sync with current global time -- injury list resolution = 1 sec.
    if(!timeSyncer_)
    {
      timeSyncer_ = scinew TimeSync(this, time_viewer_handle_);
      timeSyncer_thread_ = scinew Thread(timeSyncer_,
                       string(id+" time syncer").c_str());
    }
  }

  // get input field port
  FieldIPort *inputFieldPort = (FieldIPort *)get_iport("Input Label Volume");
  if (!inputFieldPort) {
    error("Unable to initialize input field port.");
    return;
  }
 
  // get handle to field from the input port
  if (!inputFieldPort->get(InputFieldHandle_) ||
      !InputFieldHandle_.get_rep())
  {
    remark("HotBox::execute(): No data on input field port.");
    return;
  }
  // get bounding box of input field
  inFieldBBox_ = InputFieldHandle_->mesh()->get_bounding_box();
  Point bmin, bmax;
  bmin = inFieldBBox_.min();
  bmax = inFieldBBox_.max();
  Vector inFieldBBextent = bmax - bmin;
  l2norm_ = inFieldBBextent.length() + 0.001;

  if(InputFieldHandle_->query_scalar_interface(this).get_rep() != 0)
  {
    // we have scalar data in the input field
    remark("Scalar Data on Input");
  } // end if(InputFieldHandle_->query_scalar_interface()...)
  else if(InputFieldHandle_->query_vector_interface(this).get_rep() != 0)
  {
    // we have vector data in the input field
    remark("Vector Data on Input");
  }
  else if(InputFieldHandle_->query_tensor_interface(this).get_rep() != 0)
  {
    // we have tensor data in the input field
    remark("Tensor Data on Input");
  }

  // get input Transform matrix port
  MatrixIPort *
  inputXFormMatrixPort = (MatrixIPort *)get_iport("Input Transform");
  if (!inputXFormMatrixPort) {
    error("Unable to initialize input Transform matrix port.");
  }

  // get handle to matrix from the input Transform port
  MatrixHandle inputXFormMatrixHandle;
  if(inputXFormMatrixPort != NULL && 
    (!inputXFormMatrixPort->get(inputXFormMatrixHandle) ||
     !inputXFormMatrixHandle.get_rep()))
  {
    remark("No data on input Transform matrix port.");
    inputTransform_.load_identity();
  }
  else
  {
    // get the Transform matrix data
    Matrix *xformMatrixPtr = inputXFormMatrixHandle.get_rep();
    if(xformMatrixPtr)
    {
      double minlabel;
      if(!inputXFormMatrixHandle->get_property("row_min", minlabel))
        minlabel = 0.0;
      double nrows = inputXFormMatrixHandle->nrows();
      double maxlabel;
      if (!inputXFormMatrixHandle->get_property("row_max", maxlabel))
        maxlabel = inputXFormMatrixHandle->nrows() - 1.0;

      cerr << "row_min " << minlabel << " row_max " << maxlabel;
      cerr << " nrows " << nrows << endl;

      if(!inputXFormMatrixHandle->get_property("col_min", minlabel))
      minlabel = 0.0;
      double ncols =  inputXFormMatrixHandle->ncols();
      if (!inputXFormMatrixHandle->get_property("col_max", maxlabel))
      maxlabel = inputXFormMatrixHandle->ncols() - 1.0;

      cerr << "col_min " << minlabel << " col_max " << maxlabel;
      cerr << " ncols " << ncols << endl;

      DenseMatrix *dm;
      dm = scinew DenseMatrix(inputXFormMatrixHandle->nrows(),
                              inputXFormMatrixHandle->ncols());

      for(int r = 0; r < nrows; r++)
      {
        for(int c = 0; c < ncols; c++)
        {
          (*dm)[r][c] = inputXFormMatrixHandle->get(r, c);
          cerr << (*dm)[r][c] << " ";
        }
        cerr << endl;
      }

      // expect a 4 x 4 matrix in the input Transform
      inputTransform_ = dm->toTransform();
    }
  } // end else (data on input Transform matrix port)

  // get the current selection source -- either Probe or HotBox Tcl UI
  labelIndexVal_ = 0;

  const string selectionSource(selectionsource_.get());
  if(selectionSource == "UIsetProbeLoc")
  { // move the probe to position set in UI numeric fields
    probeLoc_ = Point(gui_probeLocx_.get(),
                      gui_probeLocy_.get(),
                      gui_probeLocz_.get()
                     );
    cerr << "Probe location: " << probeLoc_<<endl;
    probeWidget_->SetPosition(probeLoc_);
  } // end else if(selectionSource == "UIsetProbeLoc")

  // get parameters from HotBox Tcl GUI
  currentSelection_ = (string)currentselection_.get();
  const int dataSource(datasource_.get());
  const string anatomyDataSrc(anatomydatasource_.get());
  const string adjacencyDataSrc(adjacencydatasource_.get());
  const string boundingBoxDataSrc(boundingboxdatasource_.get());
  const string injuryListDataSrc(injurylistdatasource_.get());
  const string enableDraw(enableDraw_.get());
  const string hipVarPath(hipvarpath_.get());

  // The segmented volume (input field to the Probe)
  // is an index into the MasterAnatomy list -- this only comes from a file
  // Read the status of this file so we can compare modification timestamps
  struct stat buf;
  if( anatomyDataSrc == "" ) {
    error("No MasterAnatomy file has been selected.  Please choose a file.");
    return;
  } else if (stat(anatomyDataSrc.c_str(), &buf)) {
    error("File '" + anatomyDataSrc + "' not found.");
    return;
  }
  if(!anatomytable_->get_num_names())
  { // label maps have not been read
    anatomytable_->readFile((char *)anatomyDataSrc.c_str());
  }
  else if(anatomyDataSrc != anatomytable_->getActiveFile())
  { // label map data source has changed
    delete anatomytable_;
    anatomytable_ = new VH_MasterAnatomy();
    anatomytable_->readFile((char *)anatomyDataSrc.c_str());
  }
  else
  {
    cout << "Master Anatomy file contains " << anatomytable_->get_num_names();
    cout << " names, max labelindex ";
    cout << anatomytable_->get_max_labelindex() << endl;
  }

  char selectName[256];
  if(selectionSource == "fromProbe")
  {
    // run the probe's functions
    executeProbe();

    if(anatomytable_->get_anatomyname(labelIndexVal_) != 0)
      strcpy(selectName, anatomytable_->get_anatomyname(labelIndexVal_));
    else
      strcpy(selectName, "unknown");
    // set currentselection in HotBox UI
    currentselection_.set(selectName);
  }
  else
  // if the selection source is from the HotBox UI -- ignore the probe
  { // get selection from HotBox Tcl GUI
    strcpy(selectName, currentSelection_.c_str());
    labelIndexVal_ = anatomytable_->get_labelindex(selectName);
  }

  if(strlen(selectName) != 0)
    cout << "VS/HotBox: selected '" << selectName << "'" << endl;
  else
    remark("Selected [NULL]");
  currentSelection_ = (string)currentselection_.get();

  if( boundingBoxDataSrc == "" ) {
    error("No Bounding Box file has been selected.  Please choose a file.");
  }

  double segVolXextent, segVolYextent, segVolZextent;
  if(!boundBoxList_ || boundingBoxDataSrc != activeBoundBoxSrc_)
  { // bounding boxes have not been read or data source has changed
    if (stat(boundingBoxDataSrc.c_str(), &buf)) {
    error("File '" + boundingBoxDataSrc + "' not found.");
    }

    if(boundBoxList_ != NULL && boundingBoxDataSrc != activeBoundBoxSrc_)
    {
      // destroy boundBoxList_
      VH_Anatomy_destroyBBox_list(boundBoxList_);
    }
    boundBoxList_ =
         VH_Anatomy_readBoundingBox_File((char *)boundingBoxDataSrc.c_str());
    activeBoundBoxSrc_ = boundingBoxDataSrc;

    // find the largest bounding volume of the segmentation
    if((maxSegmentVol_ = VH_Anatomy_findMaxBoundingBox(boundBoxList_)) != NULL)
    {
      segVolXextent = (double)
        maxSegmentVol_->get_maxX() - maxSegmentVol_->get_minX() + 0.001;
      segVolYextent = (double)
        maxSegmentVol_->get_maxY() - maxSegmentVol_->get_minY() + 0.001;
      segVolZextent = (double)
        maxSegmentVol_->get_maxZ() - maxSegmentVol_->get_minZ() + 0.001;
      maxSegBBextent_ = Point(segVolXextent, segVolYextent, segVolZextent);

      // derive factors to translate/scale segmentation bounding boxes
      // to the input labelmap field
      boxTran_ =
          Point(inFieldBBox_.min().x()-(double)maxSegmentVol_->get_minX(),
                inFieldBBox_.min().y()-(double)maxSegmentVol_->get_minY(),
                inFieldBBox_.min().z()-(double)maxSegmentVol_->get_minZ()
               );
      boxScale_.x(inFieldBBextent.x() / maxSegBBextent_.x());
      boxScale_.y(inFieldBBextent.y() / maxSegBBextent_.y());
      boxScale_.z(inFieldBBextent.z() / maxSegBBextent_.z());
    } // end if(maxSegmentVol_ != NULL)
    else
    {
      boxTran_ = Point(0.0, 0.0, 0.0);
      boxScale_ = Point(1.0, 1.0, 1.0);
    }
  } // end if(!boundBoxList_)
  VH_AnatomyBoundingBox *selectBox;
  if(boundBoxList_ != NULL)
  {
    // compare the bounding box of the input field
    // with the largest bounding volume of the segmentation
    cerr << "HotBox::execute(): input field(";
    cerr << inFieldBBox_.min() << "," << inFieldBBox_.max();
    cerr << "), extent(" << inFieldBBextent << ")" << endl;
    cerr << "                  segmentation([" << maxSegmentVol_->get_minX();
    cerr << "," << maxSegmentVol_->get_minY() << ",";
    cerr << maxSegmentVol_->get_minZ();
    cerr << "],[" << maxSegmentVol_->get_maxX() << ",";
    cerr << maxSegmentVol_->get_maxY();
    cerr << "," << maxSegmentVol_->get_maxZ() << "]), extent(";
    cerr << maxSegBBextent_ << ")" << endl;
    cerr << "Max Volume: " <<  maxSegmentVol_->get_anatomyname() << endl;

    cerr << "boxTran = " << boxTran_ << ", boxScale = " << boxScale_ << endl;

    // get the bounding box information for the selected entity
//    cerr << "TEST BEFORE: selectname: " << selectName << endl;
    selectBox = VH_Anatomy_findBoundingBox( boundBoxList_, selectName);
//    cerr << "TEST AFTER: selectname: " << selectName << endl;
//    cerr << "Anatomy: " << selectBox->get_anatomyname() << endl;
//    cerr << "Bounding Box(minX, maxX): " << selectBox->get_minX() <<"," << selectBox->get_maxX() <<endl;
//    cerr << "Bounding Box(minY, maxY): " << selectBox->get_minY() <<"," << selectBox->get_maxY() <<endl;
//    cerr << "Bounding Box(minZ, maxZ): " << selectBox->get_minZ() <<"," << selectBox->get_maxZ() <<endl;

  }
  else
  {
    selectBox = (VH_AnatomyBoundingBox *)NULL;
  }

  // we now have the anatomy name corresponding to the label value at the voxel
  if(!injListDoc_ || activeInjList_ != injuryListDataSrc)
  {
    // injury list data source or time has changed
    // Read the Injury List -- First time the HotBox Evaluates
    try {
      XMLPlatformUtils::Initialize();
    } catch (const XMLException& toCatch) {
      std::cerr << "Error during XML parser initialization! :\n"
           << StrX(toCatch.getMessage()) << endl;
      return;
    }

    // Instantiate a DOM parser for the injury list file.
    injListParser_.setDoValidation(false);

    try {
      injListParser_.parse(injuryListDataSrc.c_str());
    }  catch (const XMLException& toCatch) {
      std::cerr << "Error during parsing: '" <<
        injuryListDataSrc << "'\nException message is:  " <<
        xmlto_string(toCatch.getMessage());
        return;
    }

    // get the DOM document tree structure from the file
    injListDoc_ = injListParser_.getDocument();

    // re-populate injury list with data from new data source
    if(injured_tissue_list_.size() > 0)
    {
      for(int timeInc = 0; timeInc < injured_tissue_list_.size(); timeInc++)
      {
        injured_tissue_ = (vector <VH_injury> *)injured_tissue_list_[timeInc];
        if(injured_tissue_->size() > 0)
           injured_tissue_->clear();
      }
    }
    // parse the DOM document --
    // add an injury list per time step to the injured_tissue_ list
    parseInjuryList();

    activeInjList_ = injuryListDataSrc;
  } // end if(!injListDoc_)

  // extract the injured tissues from the DOM Document whenever time changes
  execInjuryList();

  if(currentSelection_ != lastSelection_ && dataSource == VS_DATASOURCE_OQAFMA)
  { // get the ontological hierarchy information
    fprintf(stderr, "dataSource = OQAFMA\n");

    // get the ontological hierarchy information for the current selection
    executeOQAFMA();
  } // end if(dataSource == VS_DATASOURCE_OQAFMA)
  // dataSource == FILES -- adjacency can only come from segmentation FILES
  fprintf(stderr, "dataSource = FILES[%d]\n", dataSource);
  // use fixed Adjacency Map files
  if(!adjacencytable_->get_num_names())
  { // adjacency data has not been read
    adjacencytable_->readFile((char *)adjacencyDataSrc.c_str());

    // modify the color map which corresponds to the adjacency graph
    // execSelColorMap();
  }
  else if(adjacencytable_->getActiveFile() != adjacencyDataSrc)
  { // adjacency data source has changed
    delete adjacencytable_;
    adjacencytable_ = new VH_AdjacencyMapping();
    adjacencytable_->readFile((char *)adjacencyDataSrc.c_str());

    // modify the color map which corresponds to the adjacency graph
    // execSelColorMap();
  }
  else
  {
    cout << "Adjacency Map file contains " << adjacencytable_->get_num_names();
    cout << " entries" << endl;
  } // end else(adjacency list has been read)

  string hipFileMapName;
  // the mapping from anatomical names to physiological parameters
  if( hipVarPath == "" )
  {
    error("No HIP data path has been selected.  Please choose a directory.");
  }
  else
  {
    // assume there is a file named HIP_file_map.txt in the hipVarPath
    hipFileMapName = hipVarPath + "/HIP_file_map.txt";
    if (stat(hipFileMapName.c_str(), &buf))
    {
      error("File '" + hipFileMapName + "' not found.");
    }
    if(!hipVarFileList_->get_num_names())
    { // label maps have not been read
      hipVarFileList_->readFile((char *)hipFileMapName.c_str());
    }
    else if(hipFileMapName != hipVarFileList_->getActiveFile())
    { // label map data source has changed
      delete hipVarFileList_;
      hipVarFileList_ = new VH_HIPvarMap();
      hipVarFileList_->readFile((char *)(hipFileMapName.c_str()));
    }
    else
    {
      cout << "HIP file map contains " << hipVarFileList_->get_num_names();
      cout << " names" << endl;
    }
  } // end else (HIP data path has been selected)

  // if selection or time has changed
  if(currentSelection_ != lastSelection_ ||
     lastTime_ < currentTime_ - timeEps_ ||
     currentTime_ + timeEps_ < lastTime_)
  { // execute reading the HIP time-varying physiological parameters
    // for the selected anatomical structure
    executePhysio();

    // get the surface geometry corresponding to the current selection
    executeHighlight();

    // get the adjacency info for the current selection
    // and populate the adjacency UI
    execAdjacency();
  }

  // all the time-dependent functions have been done -- update lastTime_
  if(lastTime_ < currentTime_ - timeEps_ ||
     currentTime_ + timeEps_ < lastTime_)
  {
    lastTime_ = currentTime_;
  }

  // draw HotBox Widget
  GeomGroup *HB_geomGroup = scinew GeomGroup();
  Color text_color;
  text_color = Color(1,1,1);
  MaterialHandle text_material = scinew Material(text_color);
  text_material->transparency = 0.75;

  GeomLines *lines = scinew GeomLines();
  GeomTexts *texts = scinew GeomTexts();

  VS_HotBoxUI_->setOutput(lines, texts);
  VS_HotBoxUI_->setOutMtl(text_material);

  if(enableDraw == "yes")
  { // draw HotBox UI in the Viewer
    VS_HotBoxUI_->draw(0, 0, 0.005);
  
    HB_geomGroup->add(lines);
    HB_geomGroup->add(texts);
  }

  // set output geometry port -- hotbox drawn to viewer
  GeometryOPort *HBoutGeomPort = (GeometryOPort *)get_oport("HotBox Widget");
  if(!HBoutGeomPort) {
    error("Unable to initialize HotBox Widget output geometry port.");
    return;
  }
  GeomSticky *sticky = scinew GeomSticky(HB_geomGroup);
  HBoutGeomPort->delAll();
  HBoutGeomPort->addObj( sticky, "HotBox Sticky" );
  HBoutGeomPort->flushViews();

  // set output geometry port -- Probe Widget
  if (probeWidgetid_ == -1)
  {
    GeomGroup *probeWidget_group = scinew GeomGroup;
    probeWidget_group->add(probeWidget_->GetWidget());
    GeometryOPort *
    probeOutGeomPort = (GeometryOPort *)get_oport("Probe Widget");
    if(!probeOutGeomPort)
    {
      error("Unable to initialize Probe Widget output geometry port.");
      return;
    }
    probeWidgetid_ = probeOutGeomPort->addObj(probeWidget_group,
                               "Probe Selection Widget",
                               &probeWidget_lock_);
    probeOutGeomPort->flushViews();
  } // end if (probeWidgetid_ == -1)

  // get output geometry port -- Selection Highlight
  SimpleOPort<FieldHandle> *
  highlightOutport = (SimpleOPort<FieldHandle> *)
                      getOPort("Selection Highlight");

  // send the selected field (surface) downstream
  if (!highlightOutport) {
    error("Unable to initialize selection oport.");
    return;
  }

  if(selectGeomFilehandle_.get_rep() != 0)
    highlightOutport->send(selectGeomFilehandle_);

  // get output geometry port -- Injury 0 Highlight
  SimpleOPort<FieldHandle> *
  inj0highlightOutport = (SimpleOPort<FieldHandle> *)
                      getOPort("Injury 0 Highlight");

  // send the injured field (surface) downstream
  if (!inj0highlightOutport) {
    error("Unable to initialize injury 0 oport.");
    return;
  }

  if(inj0GeomFieldhandle_.get_rep() != 0)
    inj0highlightOutport->send(inj0GeomFieldhandle_);

  // get output geometry port -- Injury 1 Highlight
  SimpleOPort<FieldHandle> *
  inj1highlightOutport = (SimpleOPort<FieldHandle> *)
                      getOPort("Injury 1 Highlight");

  // send the injured field (surface) downstream
  if (!inj1highlightOutport) {
    error("Unable to initialize injury 1 oport.");
    return;
  }

  if(inj1GeomFieldhandle_.get_rep() != 0)
    inj1highlightOutport->send(inj1GeomFieldhandle_);

  int currentTime_step = get_timeStep(currentTime_);
  // build geometry from wound icon descriptions
  makeInjGeometry();

  // get output geometry port -- Injury Icon
  SimpleOPort<FieldHandle> *
  injuryOutport = (SimpleOPort<FieldHandle> *)
                      getOPort("Injury Icon");

  // send the injury field (surface) downstream
  if (!injuryOutport) {
    error("Unable to initialize oport.");
  }
  else if(injIconFieldHandle_.get_rep() != 0)
    injuryOutport->send(injIconFieldHandle_);

  // send the NrrdData downstream
  NrrdOPort *nrrdOutPort = (NrrdOPort *)get_oport("Physiology Data");
  if(!nrrdOutPort)
  {
    error("Unable to initialize nrrdOutPort 'Physiology Data'");
    return;
  }
  if(InputNrrdHandle_.get_rep())
      nrrdOutPort->send(InputNrrdHandle_);

  if(selectBox && selectionSource == "fromHotBoxUI")
  { // set the Probe location to center of selection
    Point bmax((double)selectBox->get_maxX(),
    	       (double)selectBox->get_maxY(),
    	       (double)selectBox->get_maxZ());
    Point bmin((double)selectBox->get_minX(),
    	       (double)selectBox->get_minY(),
    	       (double)selectBox->get_minZ());
    probeLoc_ = bmin + Vector(bmax - bmin) * 0.5;
    cerr << "ProbeLoc: " << probeLoc_ <<endl;

    // scale the bounding box of the segmented region
    // to match the bounding box of the labelmap volume
    probeLoc_ = inputTransform_.project(probeLoc_);
    cerr << "Transformed ProbeLoc: " << probeLoc_ <<endl;

    probeWidget_->SetPosition(probeLoc_);

    // update probe location in the Tcl GUI
    gui_probeLocx_.set(probeLoc_.x());
    gui_probeLocy_.set(probeLoc_.y());
    gui_probeLocz_.set(probeLoc_.z());
  } // end if(selectBox && selectionSource == "fromHotBoxUI")

  // set probe scale value
  double probeScale = gui_probe_scale_.get();
  // cerr << "HotBox: probe scale: " << probeScale;
  // cerr << " * " << l2norm_ << " * 0.003 = ";
  // cerr << probeScale * l2norm_ * 0.003;
  probeWidget_->SetScale(probeScale * l2norm_ * 0.01);

  if(selectionSource != "fromProbe")
  { // clear selection source
    selectionsource_.set("fromProbe");
  }
  if(currentSelection_ != lastSelection_)
  {
    lastSelection_ = currentSelection_;
  }

} // end HotBox::execute()

char *to_HMS(char *t_str, double t)
{
    int secs = (int)t;
    int min = secs/60;
    int hr = min/60;
    sprintf(t_str, "%02d:%02d:%02d", hr, min%60, secs%60);
    return(t_str);
}

void
HotBox::sync_time(double t)
{ // time from the input to the HotBox is in seconds
    char time_str[256];
    currentTime_ = floor(t);
    
    if(lastTime_ < currentTime_ - timeEps_ ||
            currentTime_ + timeEps_ < lastTime_)
    { // time has moved forward one second
        gui_curTime_.set( to_HMS(time_str, currentTime_) );
        want_to_execute();
    }
} // end HotBox::sync_time()

/*****************************************************************************
 * method HotBox::execAdjacency()
 *****************************************************************************/
void
HotBox::execAdjacency()
{
  // get the adjacency info for the selected entity
    gui_label1_.set("-----");
    gui_label2_.set("-----");
    gui_label3_.set("-----");
    gui_label4_.set("-----");
    gui_label5_.set("-----");
    gui_label6_.set("-----");
    gui_label7_.set("-----");
    gui_label8_.set("-----");
    gui_label9_.set("-----");

  int *adjPtr;
  if((adjPtr = adjacencytable_->adjacent_to(labelIndexVal_)) == NULL)
  {
    error("HotBox::execAdjacency(): NULL pointer");
    return;
  }
  cerr << "max_labelindex: " <<  anatomytable_->get_max_labelindex() << endl;
  cerr << "num_names: " <<  anatomytable_->get_num_names() << endl;
  cerr << "labelIndexVal_: "<< labelIndexVal_ << endl;
  for (int i = 0;i < anatomytable_->get_num_names() ; i++)
  {
 //   cerr << "Adjacencies: " <<  adjPtr[i] << endl;
  }
  // fill in text labels in the HotBox
  char *adjacentName;

  if(adjacencytable_->get_num_rel(labelIndexVal_) >= 0)
  { // Note: TCL UI is 1-indexed, row-major
    if(adjPtr[0] <= anatomytable_->get_max_labelindex())
        adjacentName = anatomytable_->get_anatomyname(adjPtr[0]);
    else
    {
      cerr << "HotBox::execute(): adjacent index[" << adjPtr[0];
      cerr << "] out of range" << endl;
      // set result to "unknown"
      adjacentName = anatomytable_->get_anatomyname(0);
    }
    cerr << "HotBox::execute(): adjacent[" << adjPtr[0] << "]: ";
    cerr << adjacentName << endl;
    gui_label1_.set(adjacentName);

//    if(is_injured(adjacentName, *injured_tissue_))
//    { gui_is_injured1_.set(1); }
//    else
//    { gui_is_injured1_.set(0); }
    // OpenGL UI is indexed 0-7, column-major
    VS_HotBoxUI_->set_text(0, string(adjacentName, 0, 18));
  } // end if(adjacencytable_->get_num_rel(labelIndexVal_) >= 0)


  if(adjacencytable_->get_num_rel(labelIndexVal_) >= 1)
  { // Note: TCL UI is 1-indexed, row-major
    if(adjPtr[1] <= anatomytable_->get_max_labelindex())
        adjacentName = anatomytable_->get_anatomyname(adjPtr[1]);
    else
    {
      cerr << "HotBox::execute(): adjacent index[" << adjPtr[1];
      cerr << "] out of range" << endl;
      // set result to "unknown"
      adjacentName = anatomytable_->get_anatomyname(0);
    }
    cerr << "HotBox::execute(): adjacent[" << adjPtr[1] << "]: ";
    cerr << adjacentName << endl;
    gui_label2_.set(adjacentName);

//    if(is_injured(adjacentName, *injured_tissue_))
//    { gui_is_injured1_.set(1); }
//    else
//    { gui_is_injured1_.set(0); }
    // OpenGL UI is indexed 0-7, column-major
    VS_HotBoxUI_->set_text(0, string(adjacentName, 0, 18));
  } // end if(adjacencytable_->get_num_rel(labelIndexVal_) >= 1)

  if(adjacencytable_->get_num_rel(labelIndexVal_) >= 2)
  { // Note: TCL UI is 1-indexed, row-major
    if(adjPtr[2] <= anatomytable_->get_max_labelindex())
        adjacentName = anatomytable_->get_anatomyname(adjPtr[2]);
    else
    {
      cerr << "HotBox::execute(): adjacent index[" << adjPtr[2];
      cerr << "] out of range" << endl;
      // set result to "unknown"
      adjacentName = anatomytable_->get_anatomyname(0);
    }
    cerr << "HotBox::execute(): adjacent[" << adjPtr[2] << "]: ";
    cerr << adjacentName << endl;
    gui_label3_.set(adjacentName);

//    if(is_injured(adjacentName, *injured_tissue_))
//    { gui_is_injured2_.set(1); }
//    else
//    { gui_is_injured2_.set(0); }
    // OpenGL UI is indexed 0-7, column-major
    VS_HotBoxUI_->set_text(3, string(adjacentName, 0, 18));
  } // end if(adjacencytable_->get_num_rel(labelIndexVal_) >= 2)


  if(adjacencytable_->get_num_rel(labelIndexVal_) >= 3)
  {  // Note: TCL UI is 1-indexed, row-major
    if(adjPtr[3] <= anatomytable_->get_max_labelindex())
        adjacentName = anatomytable_->get_anatomyname(adjPtr[3]);
    else
    {
      cerr << "HotBox::execute(): adjacent index[" << adjPtr[3];
      cerr << "] out of range" << endl;
      // set result to "unknown"
      adjacentName = anatomytable_->get_anatomyname(0);
    }
    cerr << "HotBox::execute(): adjacent[" << adjPtr[3] << "]: ";
    cerr << adjacentName << endl;
    gui_label4_.set(adjacentName);

//    if(is_injured(adjacentName, *injured_tissue_))
//    { gui_is_injured3_.set(1); }
//    else
//    { gui_is_injured3_.set(0); }
    // OpenGL UI is indexed 0-7, column-major
    VS_HotBoxUI_->set_text(5, string(adjacentName, 0, 18));
  } // end if(adjacencytable_->get_num_rel(labelIndexVal_) >= 3)


  string selectName = currentselection_.get();
  gui_label5_.set(selectName);
 
  if(adjacencytable_->get_num_rel(labelIndexVal_) >= 4)
  { // Note: TCL UI is 1-indexed, row-major
    if(adjPtr[4] <= anatomytable_->get_max_labelindex())
        adjacentName = anatomytable_->get_anatomyname(adjPtr[4]);
    else
    {
      cerr << "HotBox::execute(): adjacent index[" << adjPtr[4];
      cerr << "] out of range" << endl;
      // set result to "unknown"
      adjacentName = anatomytable_->get_anatomyname(0);
    }
    cerr << "HotBox::execute(): adjacent[" << adjPtr[4] << "]: ";
    cerr << adjacentName << endl;
    gui_label6_.set(adjacentName);
                                                                                              
//    if(is_injured(adjacentName, *injured_tissue_))
//    { gui_is_injured7_.set(1); }
//    else
//    { gui_is_injured7_.set(0); }
    // OpenGL UI is 0-indexed, column-major
    VS_HotBoxUI_->set_text(2, string(adjacentName, 0, 18));
  } // end if(adjacencytable_->get_num_rel(labelIndexVal_) >= 4)

 
  if(adjacencytable_->get_num_rel(labelIndexVal_) >= 5)
  { // Note: TCL UI is 1-indexed, row-major
    if(adjPtr[5] <= anatomytable_->get_max_labelindex())
        adjacentName = anatomytable_->get_anatomyname(adjPtr[5]);
    else
    {
      cerr << "HotBox::execute(): adjacent index[" << adjPtr[5];
      cerr << "] out of range" << endl;
      // set result to "unknown"
      adjacentName = anatomytable_->get_anatomyname(0);
    }
    cerr << "HotBox::execute(): adjacent[" << adjPtr[5] << "]: ";
    cerr << adjacentName << endl;
    gui_label7_.set(adjacentName);

//    if(is_injured(adjacentName, *injured_tissue_))
//    { gui_is_injured7_.set(1); }
//    else
//    { gui_is_injured7_.set(0); }
    // OpenGL UI is 0-indexed, column-major
    VS_HotBoxUI_->set_text(2, string(adjacentName, 0, 18));
  } // end if(adjacencytable_->get_num_rel(labelIndexVal_) >= 5)

  if(adjacencytable_->get_num_rel(labelIndexVal_) >= 6)
  { // Note: TCL UI is 1-indexed, row-major 
    if(adjPtr[6] <= anatomytable_->get_max_labelindex())
        adjacentName = anatomytable_->get_anatomyname(adjPtr[6]);
    else
    {
      cerr << "HotBox::execute(): adjacent index[" << adjPtr[6];
      cerr << "] out of range" << endl;
      // set result to "unknown"
      adjacentName = anatomytable_->get_anatomyname(0);
    }
    cerr << "HotBox::execute(): adjacent[" << adjPtr[6] << "]: ";
    cerr << adjacentName << endl;
    gui_label8_.set(adjacentName);

//    if(is_injured(adjacentName, *injured_tissue_))
//    { gui_is_injured8_.set(1); }
//    else
//    { gui_is_injured8_.set(0); }
    // OpenGL UI is indexed 0-7, column-major
    VS_HotBoxUI_->set_text(4, string(adjacentName, 0, 18));
  } // end if(adjacencytable_->get_num_rel(labelIndexVal_) >= 6)

  if(adjacencytable_->get_num_rel(labelIndexVal_) >= 7)
  { // Note: TCL UI is 1-indexed, row-major
    if(adjPtr[7] <= anatomytable_->get_max_labelindex())
        adjacentName = anatomytable_->get_anatomyname(adjPtr[7]);
    else
    {
      cerr << "HotBox::execute(): adjacent index[" << adjPtr[7];
      cerr << "] out of range" << endl;
      // set result to "unknown"
      adjacentName = anatomytable_->get_anatomyname(0);
    }
    cerr << "HotBox::execute(): adjacent[" << adjPtr[7] << "]: ";
    cerr << adjacentName << endl;
    gui_label9_.set(adjacentName);

//    if(is_injured(adjacentName, *injured_tissue_))
//    { gui_is_injured9_.set(1); }
//    else
//    { gui_is_injured9_.set(0); }
    // OpenGL UI is indexed 0-7, column-major
    VS_HotBoxUI_->set_text(7, string(adjacentName, 0, 18));
  } // end if(adjacencytable_->get_num_rel(labelIndexVal_) >= 7)


} // end HotBox::execAdjacency()

/*****************************************************************************
 * method HotBox::executeOQAFMA()
 *****************************************************************************/

void
HotBox::executeOQAFMA()
{ // get the ontological hierarchy information
  char *oqafma_relation[VH_LM_NUM_NAMES];
  char partsName[256], capitalName[256], selectName[256];
  int num_struQLret;

  // skip execution if current selection has not changed
  if(currentSelection_ == lastSelection_) return;

  ns1__processStruQLResponse resultStruQL;
  ServiceInterfaceSoapBinding ws;

  // set the Web Services URL
  const string oqafmaDataSrc(oqafmadatasource_.get());
  if(strcmp(ws.endpoint, oqafmaDataSrc.c_str()))
  { // OQAFMA Web services URL has changed
    ws.endpoint = strdup(oqafmaDataSrc.c_str());
  }

  /////////////////////////////////////////////////////////////////////////////
  // get the hierarchical children of the selection from the FMA
  // build the query to send to OQAFMA
  std::string p2;
  const int queryType(querytype_.get());
  switch(queryType)
  {
    case VS_QUERYTYPE_ADJACENT_TO:
    case VS_QUERYTYPE_CONTAINS:
    {
      p2 = "WHERE X->\":NAME\"->\"";
      p2 += capitalize(capitalName, selectName);
      p2 += "\", X->\"part\"+.\"contains\"->Y, Y->\":NAME\"->Contains CREATE The";
      p2 += space_to_underbar(partsName, capitalName);
      p2 += "(Contains)";
      break;
    }
    case VS_QUERYTYPE_PARTS:
    {
      p2 = "WHERE X->\":NAME\"->\"";
      p2 += capitalize(capitalName, selectName);
      p2 += "\", X->\"part\"+->Y, Y->\":NAME\"->Parts CREATE The";
      p2 += space_to_underbar(partsName, capitalName);
      p2 += "(Parts)";
      break;
    }
    case VS_QUERYTYPE_PARTCONTAINS:
    default:
    {
    }
  }
  cout << "OQAFMA query: " <<  p2  << endl;
  // launch a query via OQAFMA/Protege C function calls
  if (ws.ns1__processStruQL(p2, resultStruQL) != SOAP_OK)
  {
    soap_print_fault(ws.soap, stderr);
  }
  else
  {
    // parse XML query results
    std::string OQAFMA_result = resultStruQL._processStruQLReturn;
    cout << OQAFMA_result;

    // Instantiate a DOM parser for the OQAFMA query results
    XercesDOMParser struQLretParser;
    struQLretParser.setDoValidation(false);
  
    // create a Xerces InputSource to hold the query results
     MemBufInputSource StruQLRetInputSrc (
          (const XMLByte*)OQAFMA_result.c_str(),
          strlen(OQAFMA_result.c_str()),
          "HotBoxStruQL", false);
    try {
      struQLretParser.parse(StruQLRetInputSrc);
    }  catch (const XMLException& toCatch) {
      std::cerr << "Error during parsing: StruQLReturn\n"
                << "Exception message is:  " <<
        xmlto_string(toCatch.getMessage());
        return;
    }

    DOMDocument *struQLretDoc = struQLretParser.getDocument();
    DOMNodeList *struQLretList;
    const XMLCh* xs;
    switch(queryType)
    {
      case VS_QUERYTYPE_ADJACENT_TO:
      case VS_QUERYTYPE_CONTAINS:
      {
        xs = to_xml_ch_ptr("Contains");
        break;
      }
      case VS_QUERYTYPE_PARTS:
      {
        xs = to_xml_ch_ptr("Parts");
        break;
      }
      case VS_QUERYTYPE_PARTCONTAINS:
      default:
      {
      }
    } // end switch(queryType)
    struQLretList = struQLretDoc->getElementsByTagName(xs);

    num_struQLret = struQLretList->getLength();
    if (num_struQLret == 0) {
      cout << "HotBox.cc: no entities in StruQL return" << endl;
    }
    else
    {
      cout << "HotBox.cc: " << num_struQLret
           << " entities in StruQL return" << endl;
      if(num_struQLret >= VH_LM_NUM_NAMES)
         num_struQLret = VH_LM_NUM_NAMES;
      for (int i = 0;i < num_struQLret; i++)
      {
        if (!(struQLretList->item(i)))
        {
          cout << "Error: NULL DOM node" << std::endl;
          continue;
        }
        DOMNode &node = *(struQLretList->item(i));
        // debugging...
        // cout << "Node[" << i << "] type: " << node.getNodeType();
        // cout << " name: " << to_char_ptr(node.getNodeName()) << " ";
        if(node.hasChildNodes())
        {
            // cout << " has child nodes " << endl;
            DOMNode  *elem = node.getFirstChild();
            if(elem == 0)
                cout << " cannot get first child" << endl;
            else
            {
                // cout << " element: "
                //     << to_char_ptr(elem->getNodeValue()) << endl;
                oqafma_relation[i] =
                     strdup(to_char_ptr(elem->getNodeValue()));
            }
        }
        else
            cout << " has no child nodes" << endl;
      } // end for (i = 0;i < num_struQLret; i++)
    } // end else (num_struQLret != 0)
  } // end else (SOAP_OK)
  // catch(exception& e)
  // {
  //   printf("Unknown exception has occured\n");
  // }
  // catch(...)
  // {
  //   printf("Unknown exception has occured\n");
  // }

  // set vars in HotBox Tcl GUI
  gui_sibling0_.set(selectName);
  if(num_struQLret > 0)
     gui_child0_.set(oqafma_relation[0]);
  else
     gui_child0_.set("");
  if(num_struQLret > 1)
     gui_child1_.set(oqafma_relation[1]);
  else
     gui_child1_.set("");
  if(num_struQLret > 2)
     gui_child2_.set(oqafma_relation[2]);
  else
     gui_child2_.set("");
  if(num_struQLret > 3)
     gui_child3_.set(oqafma_relation[3]);
  else
     gui_child3_.set("");
  if(num_struQLret > 4)
     gui_child4_.set(oqafma_relation[4]);
  else
     gui_child4_.set("");
  if(num_struQLret > 5)
     gui_child5_.set(oqafma_relation[5]);
  else
     gui_child5_.set("");
  if(num_struQLret > 6)
     gui_child6_.set(oqafma_relation[6]);
  else
     gui_child6_.set("");
  if(num_struQLret > 7)
     gui_child7_.set(oqafma_relation[7]);
  else
     gui_child7_.set("");
  if(num_struQLret > 8)
     gui_child8_.set(oqafma_relation[8]);
  else
     gui_child8_.set("");
  if(num_struQLret > 9)
     gui_child9_.set(oqafma_relation[9]);
  else
     gui_child9_.set("");
  if(num_struQLret > 10)
     gui_child10_.set(oqafma_relation[10]);
  else
     gui_child10_.set("");
  if(num_struQLret > 11)
     gui_child11_.set(oqafma_relation[11]);
  else
     gui_child11_.set("");
  if(num_struQLret > 12)
     gui_child12_.set(oqafma_relation[12]);
  else
     gui_child12_.set("");
  if(num_struQLret > 13)
     gui_child13_.set(oqafma_relation[13]);
  else
     gui_child13_.set("");
  if(num_struQLret > 14)
     gui_child14_.set(oqafma_relation[14]);
  else
     gui_child14_.set("");
  if(num_struQLret > 15)
     gui_child15_.set(oqafma_relation[15]);
  else
     gui_child15_.set("");

  //////////////////////////////////////////////////////////////////////////
  // get the hierarchical parent of the selection from the FMA
  // build the query to send to OQAFMA
  switch(queryType)
  {
    case VS_QUERYTYPE_ADJACENT_TO:
    case VS_QUERYTYPE_CONTAINS:
    {
      p2 = "WHERE X->\":NAME\"->\"";
      p2 += capitalize(capitalName, selectName);
      p2 += "\", Y->\"part\"+.\"contains\"->X, Y->\":NAME\"->Parent CREATE The";
      p2 += space_to_underbar(partsName, capitalName);
      p2 += "(Parent)";
      break;
    }
    case VS_QUERYTYPE_PARTS:
    {
      p2 = "WHERE X->\":NAME\"->\"";
      p2 += capitalize(capitalName, selectName);
      p2 += "\", Y->\"part\"+->X, Y->\":NAME\"->Parent CREATE The";
      p2 += space_to_underbar(partsName, capitalName);
      p2 += "(Parent)";
      break;
    }
    case VS_QUERYTYPE_PARTCONTAINS:
    default:
    {
    }
  }
  cout << "OQAFMA query: " <<  p2  << endl;
  // launch a query via OQAFMA/Protege C function calls
  if (ws.ns1__processStruQL(p2, resultStruQL) != SOAP_OK)
  {
    soap_print_fault(ws.soap, stderr);
  }
  else
  {
    // parse XML query results
    std::string OQAFMA_result = resultStruQL._processStruQLReturn;
    cout << OQAFMA_result;

    // Instantiate a DOM parser for the OQAFMA query results
    XercesDOMParser struQLretParser;
    struQLretParser.setDoValidation(false);
  
    // create a Xerces InputSource to hold the query results
     MemBufInputSource StruQLRetInputSrc (
          (const XMLByte*)OQAFMA_result.c_str(),
          strlen(OQAFMA_result.c_str()),
          "HotBoxStruQL", false);
    try {
      struQLretParser.parse(StruQLRetInputSrc);
    }  catch (const XMLException& toCatch) {
      std::cerr << "Error during parsing: StruQLReturn\n"
                << "Exception message is:  " <<
        xmlto_string(toCatch.getMessage());
        return;
    }

    DOMDocument *struQLretDoc = struQLretParser.getDocument();
    DOMNodeList *struQLretList;
    const XMLCh* xs;
    switch(queryType)
    {
      case VS_QUERYTYPE_ADJACENT_TO:
      case VS_QUERYTYPE_CONTAINS:
      case VS_QUERYTYPE_PARTS:
      {
        xs = to_xml_ch_ptr("Parent");
        break;
      }
      case VS_QUERYTYPE_PARTCONTAINS:
      default:
      {
      }
    } // end switch(queryType)
    struQLretList = struQLretDoc->getElementsByTagName(xs);

    num_struQLret = struQLretList->getLength();
    if (num_struQLret == 0) {
      cout << "HotBox.cc: no entities in StruQL return" << endl;
    }
    else
    {
      cout << "HotBox.cc: " << num_struQLret
           << " entities in StruQL return" << endl;
      if(num_struQLret >= VH_LM_NUM_NAMES)
         num_struQLret = VH_LM_NUM_NAMES;
      for (int i = 0;i < num_struQLret; i++)
      {
        if (!(struQLretList->item(i)))
        {
          cout << "Error: NULL DOM node" << std::endl;
          continue;
        }
        DOMNode &node = *(struQLretList->item(i));
        // debugging...
        // cout << "Node[" << i << "] type: " << node.getNodeType();
        // cout << " name: " << to_char_ptr(node.getNodeName()) << " ";
        if(node.hasChildNodes())
        {
            // cout << " has child nodes " << endl;
            DOMNode  *elem = node.getFirstChild();
            if(elem == 0)
                cout << " cannot get first child" << endl;
            else
            {
                // cout << " element: "
                //     << to_char_ptr(elem->getNodeValue()) << endl;
                oqafma_relation[i] =
                     strdup(to_char_ptr(elem->getNodeValue()));
            }
        }
        else
            cout << " has no child nodes" << endl;
      } // end for (i = 0;i < num_struQLret; i++)
    } // end else (num_struQLret != 0)
  } // end else (SOAP_OK)
  // catch(exception& e)
  // {
  //   printf("Unknown exception has occured\n");
  // }
  // catch(...)
  // {
  //   printf("Unknown exception has occured\n");
  // }

  if(num_struQLret > 0)
     gui_parent0_.set(oqafma_relation[0]);
  else
     gui_parent0_.set("");
  if(num_struQLret > 1)
     gui_parent1_.set(oqafma_relation[1]);
  else
     gui_parent1_.set("");
  if(num_struQLret > 2)
     gui_parent2_.set(oqafma_relation[2]);
  else
     gui_parent2_.set("");
  if(num_struQLret > 3)
     gui_parent3_.set(oqafma_relation[3]);
  else
     gui_parent3_.set("");
  if(num_struQLret > 4)
     gui_parent4_.set(oqafma_relation[4]);
  else
     gui_parent4_.set("");
  if(num_struQLret > 5)
     gui_parent5_.set(oqafma_relation[5]);
  else
     gui_parent5_.set("");
  if(num_struQLret > 6)
     gui_parent6_.set(oqafma_relation[6]);
  else
     gui_parent6_.set("");
  if(num_struQLret > 7)
     gui_parent7_.set(oqafma_relation[7]);
  else
     gui_parent7_.set("");

  // **** magic occurs here **** //
  // grab control of the Tcl GUI program asynchronously
  std::string tclResult;
  gui->lock();
  // re-populate Hierarchy Browser lists with array members
  std::string
   evalStr = "set " + gui_parent_list_.get() + " [list [set " + 
             gui_name_.get() + "-gui_parent(0)] [set " +
             gui_name_.get() + "-gui_parent(1)] [set " +
             gui_name_.get() + "-gui_parent(2)] [set " +
             gui_name_.get() + "-gui_parent(3)] [set " +
             gui_name_.get() + "-gui_parent(4)] [set " +
             gui_name_.get() + "-gui_parent(5)] [set " +
             gui_name_.get() + "-gui_parent(6)] [set " +
             gui_name_.get() + "-gui_parent(7)]]";
  cerr << "gui->eval(" << evalStr << ")" << endl;
  gui->eval(evalStr, tclResult);
  istringstream iss(tclResult);
  cerr << iss;

  gui->eval("set " + gui_sibling_list_.get() + " [list [set " +
            gui_name_.get() + "-gui_sibling(0)] [set " +
            gui_name_.get() + "-gui_sibling(1)] [set " +
            gui_name_.get() + "-gui_sibling(2)] [set " +
            gui_name_.get() + "-gui_sibling(3)]]", tclResult);

  gui->eval("set " + gui_child_list_.get() + " [list [set " +
            gui_name_.get() + "-gui_child(0)] [set " +
            gui_name_.get() + "-gui_child(1)] [set " +
            gui_name_.get() + "-gui_child(2)] [set " +
            gui_name_.get() + "-gui_child(3)] [set " +
            gui_name_.get() + "-gui_child(4)] [set " +
            gui_name_.get() + "-gui_child(5)] [set " +
            gui_name_.get() + "-gui_child(6)] [set " +
            gui_name_.get() + "-gui_child(7)] [set " +
            gui_name_.get() + "-gui_child(8)] [set " +
            gui_name_.get() + "-gui_child(9)] [set " +
            gui_name_.get() + "-gui_child(10)] [set " +
            gui_name_.get() + "-gui_child(11)] [set " +
            gui_name_.get() + "-gui_child(12)] [set " +
            gui_name_.get() + "-gui_child(13)] [set " +
            gui_name_.get() + "-gui_child(14)] [set " +
            gui_name_.get() + "-gui_child(15)]]", tclResult);

  gui->unlock();

  // clean up
  for (int i = 0; i < num_struQLret; i++)
  {
    if(oqafma_relation[i] != 0)
    {
      free(oqafma_relation[i]);
      oqafma_relation[i] = 0;
    }
  }
  num_struQLret = 0;
} // end HotBox::executeOQAFMA()

/*****************************************************************************
 * method HotBox::parseInjuryList()/traverseDOMtree()
  // walk the DOM document collecting information on injured tissues
  // <event>
  // <wound woundName="Left ventricular penetration" woundID="1.0">
  //     <timeStamp time="1.0" unit="s"/>
  //     <diagnosis>tamponade</diagnosis>
  //     <primaryInjuryList>
  //         <injuryEntity injuryName="Ablated LV myocardium" injuryID="1.1" >
  //             <ablateRegion>
  //             <pathEntity PATID="TBD" PATname="Percent">
  //               <fmaEntity FMAname="Wall of left ventricle" FMAID="9556"/>
  //               <probability="...">
  //               <geometry>
  //                 <dimEntity DIMID="cylinder" DIMname="cylinder">
  //                     <pathEntity PATname="Location-3D" PATID="TBD">
  //                         <label>Axis end point</label>
  //                         <value>400., 250., 1460.</value>
  //                         <unit>mm</unit>
  //                     </patEntity>
  //                     <patEntity PATname="Location-3D" PATID="TBD">
  //                         <label>Axis start point</label>
  //                         <value>400., 250., 1460.</value>
  //                         <unit>mm</unit>
  //                     </patEntity>
  //                     <patEntity PATname="Length" PATID="TBD">
  //                         <label>diameter</label>
  //                         <value>200.0</value>
  //                         <unit>mm</unit>^M
  //                     </patEntity>
  //                 </dimEntity>
  //               </geometry>
  /             </pathEntity>
  //         </ablateRegion>
  //         </injuryEntity>
  //     </primaryInjuryList>
  // </wound>
  // </event>
 *****************************************************************************/
void
HotBox::traverseDOMtree(DOMNode &woundNode, int nodeIndex, double *curTime,
                        VH_injury **injuryPtr)
{
  // debugging...
  if(!strcmp(to_char_ptr(woundNode.getNodeName()), "timeStamp") ||
     !strcmp(to_char_ptr(woundNode.getNodeName()), "diagnosis") ||
     !strcmp(to_char_ptr(woundNode.getNodeName()), "probability") ||
     !strcmp(to_char_ptr(woundNode.getNodeName()), "primaryInjuryList") ||
     !strcmp(to_char_ptr(woundNode.getNodeName()), "secondaryInjuryList") ||
     !strcmp(to_char_ptr(woundNode.getNodeName()), "ablateRegion") ||
     !strcmp(to_char_ptr(woundNode.getNodeName()), "stunRegion") ||
     !strcmp(to_char_ptr(woundNode.getNodeName()), "fmaEntity") ||
     !strcmp(to_char_ptr(woundNode.getNodeName()), "dimEntity")
    )
  {
    cout << "Node[" << nodeIndex << "] type: " << woundNode.getNodeType();
    cout << " name: " << to_char_ptr(woundNode.getNodeName());
    cout << " value: " << to_char_ptr(woundNode.getNodeValue()) << endl;
  }

  // key on node name
  if(!strcmp(to_char_ptr(woundNode.getNodeName()), "label"))
  {
    // check context for geometry
    if((*injuryPtr)->isGeometry)
    {
      // get the value of the label
      string geomParamName =
              string(to_char_ptr(woundNode.getFirstChild()->getNodeValue()));
      // debugging...
      // cout << geomParamName << endl;

      if(geomParamName == string("Axis start point"))
      {
        (*injuryPtr)->context = SET_AXIS_START_POINT;
      }
      else if(geomParamName == string("Axis end point"))
      {
        (*injuryPtr)->context = SET_AXIS_END_POINT;
      }
      else if(geomParamName == string("Diameter") ||
              geomParamName == string("diameter"))
      {
        (*injuryPtr)->context = SET_DIAMETER;
      }
      else if(geomParamName == string("inside diameter"))
      {
        (*injuryPtr)->context = SET_INSIDE_DIAMETER;
      }
      else
      { // unset context
        (*injuryPtr)->context = UNSET;
        cerr << "Unknown label: " << geomParamName << endl;
      }
    } // end if((*injuryPtr)->isGeometry)
  } // end if(woundNode.getNodeName() == "label")
  else if(!strcmp(to_char_ptr(woundNode.getNodeName()), "value"))
  {
    // check context for geometry
    if((*injuryPtr)->isGeometry)
    {
      // fill in the value for the correct context
      char *geomValueStr = (char *)
              to_char_ptr(woundNode.getFirstChild()->getNodeValue());
      // debugging...
      // cout << geomValueStr << endl;
      float x, y, z, diam;

      if( (*injuryPtr)->context == SET_AXIS_START_POINT)
      { // expect a float triple
        if(sscanf(geomValueStr, "%f, %f, %f", &x, &y, &z) != 3)
          cerr << "Error reading Axis Start Point" << endl;
        (*injuryPtr)->axisX0 = x;
        (*injuryPtr)->axisY0 = y;
        (*injuryPtr)->axisZ0 = z;
        (*injuryPtr)->point0set = true;
      }
      else if( (*injuryPtr)->context == SET_AXIS_END_POINT)
      { // expect a float triple
        if(sscanf(geomValueStr, "%f, %f, %f", &x, &y, &z) != 3)
          cerr << "Error reading Axis End Point" << endl;
        (*injuryPtr)->axisX1 = x;
        (*injuryPtr)->axisY1 = y;
        (*injuryPtr)->axisZ1 = z;
        (*injuryPtr)->point1set = true;
      }
      else if( (*injuryPtr)->context == SET_DIAMETER)
      { // expect a single float
        if(sscanf(geomValueStr, "%f", &diam) != 1)
          cerr << "Error reading diameter" << endl;
        (*injuryPtr)->rad0 = (*injuryPtr)->rad1 = diam/2.0;
        (*injuryPtr)->rad0set = (*injuryPtr)->rad1set = true;
      }
      else if( (*injuryPtr)->context == SET_INSIDE_DIAMETER)
      { // expect a single float
        if(sscanf(geomValueStr, "%f", &diam) != 1)
          cerr << "Error reading inside diameter" << endl;
        (*injuryPtr)->inside_rad0 = (*injuryPtr)->inside_rad1 = diam/2.0;
        (*injuryPtr)->inside_rad0set = (*injuryPtr)->inside_rad1set = true;
      }
      else
      {
        cerr << "Bad Context" << endl;
      }
    } // end if((*injuryPtr)->isGeometry)
  } // end if(woundNode.getNodeName() == "value")
  else if(!strcmp(to_char_ptr(woundNode.getNodeName()),
          "primaryInjuryList"))
  {
    (*injuryPtr)->isPrimaryInjury = true;
  }
  else if(!strcmp(to_char_ptr(woundNode.getNodeName()),
          "secondaryInjuryList"))
  {
    (*injuryPtr)->isSecondaryInjury = true;
  }
  else if(!strcmp(to_char_ptr(woundNode.getNodeName()),
          "ablateRegion"))
  {
    (*injuryPtr)->isAblate = true;
  }
  else if(!strcmp(to_char_ptr(woundNode.getNodeName()),
          "stunRegion"))
  {
    (*injuryPtr)->isStun = true;
  }
  else if(!strcmp(to_char_ptr(woundNode.getNodeName()),
          "diagnosis"))
  {
    (*injuryPtr)->diagnosis = string(to_char_ptr(woundNode.getNodeValue()));
  }
  // get attributes
  if(woundNode.hasAttributes())
  {
    int num_attr = woundNode.getAttributes()->getLength();
    for(int i = 0; i < num_attr; i++)
    {
      DOMNode *
      elem = woundNode.getAttributes()->item(i);

      // debugging...
      if(!strcmp(to_char_ptr(woundNode.getNodeName()), "timeStamp") ||
         !strcmp(to_char_ptr(woundNode.getNodeName()), "diagnosis") ||
         !strcmp(to_char_ptr(woundNode.getNodeName()), "probability") ||
         !strcmp(to_char_ptr(woundNode.getNodeName()), "fmaEntity") ||
         !strcmp(to_char_ptr(woundNode.getNodeName()), "dimEntity")
        )
      {
        cout << " attr name: " << to_char_ptr(elem->getNodeName());
        cout << " value: " << to_char_ptr(elem->getNodeValue()) << endl;
      }

      if(!strcmp(to_char_ptr(woundNode.getNodeName()), "timeStamp") &&
         !strcmp(to_char_ptr(elem->getNodeName()), "time"))
      {
        (*injuryPtr)->timeStamp = atof(to_char_ptr(elem->getNodeValue()));
        // collect the wound for the current timeStep
        (*injuryPtr)->timeSet = true;
      }
      else if(!strcmp(to_char_ptr(woundNode.getNodeName()), "timeStamp") &&
         !strcmp(to_char_ptr(elem->getNodeName()), "unit"))
      {
        (*injuryPtr)->timeUnit = to_char_ptr(elem->getNodeValue());
      }
      else if(!strcmp(to_char_ptr(woundNode.getNodeName()), "probability") &&
         !strcmp(to_char_ptr(elem->getNodeName()), "prop"))
      {
        (*injuryPtr)->probability = atof(to_char_ptr(elem->getNodeValue()));
      }
      else if(!strcmp(to_char_ptr(woundNode.getNodeName()), "fmaEntity") &&
              !strcmp(to_char_ptr(elem->getNodeName()), "FMAname"))
      {
        (*injuryPtr)->anatomyname = string(to_char_ptr(elem->getNodeValue()));
        (*injuryPtr)->nameSet = true;
      }
      else if(!strcmp(to_char_ptr(woundNode.getNodeName()), "dimEntity") &&
              !strcmp(to_char_ptr(elem->getNodeName()), "DIMname"))
      {
        (*injuryPtr)->geom_type = string(to_char_ptr(elem->getNodeValue()));
        (*injuryPtr)->isGeometry = true;
      }
    } // end for(int i = 0; i < num_attr; i++)
  } // end if(woundNode.hasAttributes())

  // if this node is complete
  if((*injuryPtr)->iscomplete())
  { // add this node to the injured tissue list

    cerr << "Adding: " << endl;
    (*injuryPtr)->print();
    injured_tissue_->push_back(**injuryPtr);
    double woundTime = (*injuryPtr)->timeStamp;
    string woundUnit = (*injuryPtr)->timeUnit;
    // global clock is in seconds -- convert wound timeStamp to match
    if((*injuryPtr)->timeUnit == "min")
        *curTime = woundTime * 60.0;
    else // units default to seconds
        *curTime = woundTime;
    // create the next injury record
    *injuryPtr = new VH_injury();

    // carry through timeStamp and units
    (*injuryPtr)->timeStamp =  woundTime;
    (*injuryPtr)->timeUnit =  woundUnit;
    (*injuryPtr)->timeSet = true;
  }
  if(woundNode.hasChildNodes())
  {
    DOMNodeList *
    woundChildList = woundNode.getChildNodes();
    int num_woundChildList = woundChildList->getLength();
    // traverse the children of this node
    for(int i = 0; i < num_woundChildList; i++)
    {
      DOMNode &woundChild = *(woundChildList->item(i));
      // recurse down the tree
      traverseDOMtree(woundChild, i, curTime, injuryPtr);
    } // end for(int i = 0; i < num_woundChildList; i++)
  } // end if(woundNode.hasChildNodes())
} // end HotBox::traverseDOMtree()

void
HotBox::parseInjuryList()
{
  double last_timeStep;
  double cur_timeStep;
  double min_timeStep = HUGE;
  double max_timeStep = -HUGE;
  double timeIncr = 0.0;

  // create the first injury record
  VH_injury *injuryPtr = new VH_injury();

  // create the injury list for the first time step
  injured_tissue_ = new vector <VH_injury>;

  // parse the DOM document
  DOMNodeList *
  woundList = injListDoc_->getElementsByTagName(to_xml_ch_ptr("wound"));
  int num_woundList = woundList->getLength();
  const string injuryListDataSrc(injurylistdatasource_.get());

  // we have to assume one wound entity per timeStep
  // however, timeStamps do not always start at 0
  // and may not be integers

  if (num_woundList <= 0)
  {
    cout << "HotBox.cc: no wounded entities in Injury List" << endl;
  }
  else
  {
    cout << "HotBox.cc: xml file: " << injuryListDataSrc << ": "
         << num_woundList << " wounded region entities" << endl;

    // for each time step
    for(int i = 0; i < num_woundList; i++)
    {
      if (!(woundList->item(i)))
      {
        std::cerr << "Error: NULL DOM node" << std::endl;
        continue;
      }
      DOMNode &
      woundNode = *(woundList->item(i));
      // traverse the DOM document tree
      traverseDOMtree(woundNode, i, &cur_timeStep, &injuryPtr);

      // compute min, max timeStep, avg timeIncr
      if(cur_timeStep >= max_timeStep)
        max_timeStep = cur_timeStep;
      if(cur_timeStep <= min_timeStep)
        min_timeStep = cur_timeStep;
      if(i == 0) last_timeStep = cur_timeStep;
      double deltaT = cur_timeStep - last_timeStep;
      timeIncr += deltaT;
      last_timeStep = cur_timeStep;

      // report number of injuries read
      cerr << "HotBox::parseInjuryList(): time step[" << cur_timeStep << "] ";
      cerr << injured_tissue_->size() << " injuries found" << endl;

      // add the injury list for this timestep
      injured_tissue_list_.push_back(injured_tissue_);
      // allocate the next injury list
      injured_tissue_ = new vector <VH_injury>;
    } // end for(int i = 0; i < num_woundList; i++)
  } // end else (num_woundList > 0)
  // initialize the injury list to timeStep 0
  injured_tissue_ = (vector <VH_injury> *)injured_tissue_list_[0];

  // compute average time increment, epsilon
  timeIncr = timeIncr / (double)num_woundList;
  timeEps_ = timeIncr/4.0;

  cerr << "HotBox::parseInjuryList(): " << injured_tissue_list_.size();
  cerr << " injury events found, time range[" << min_timeStep << ", ";
  cerr << max_timeStep<< "] increment " << timeIncr << endl;

} // end parseInjuryList()

/*****************************************************************************
 method: get_timeStep()

 Description: Return the integer index of the injury list containing the
              timeStamp matching the input target.
 *****************************************************************************/
 
int
HotBox::get_timeStep(double targ_timeStamp)
{
  int retIndex = -1;
  // units default to seconds
  double wound_timeStamp = targ_timeStamp;
  vector <VH_injury> *saveInjury = injured_tissue_;

  // debugging...
  cerr << "HotBox::get_timeStep(" << targ_timeStamp << ")" << endl;
  cerr << "	" << injured_tissue_list_.size() << " timeSteps" << endl;
  for(int i = 0; i < injured_tissue_list_.size(); i++)
  {
    injured_tissue_ =
          (vector <VH_injury> *)injured_tissue_list_[i];
    // debugging...
    // cerr << " injured_tissue_list_[" << i << "] size: ";
    // cerr << injured_tissue_->size();
    if(injured_tissue_ && injured_tissue_->size() > 0)
    {
      VH_injury woundPtr = (*injured_tissue_)[0];
      // debugging...
      // cerr << " timeStamp " << woundPtr.timeStamp;
      // target timeStamp is in seconds -- convert wound timeStamp to match
      if(woundPtr.timeUnit == "min")
         wound_timeStamp = woundPtr.timeStamp * 60.0;
      else // default to seconds
         wound_timeStamp = woundPtr.timeStamp;

      if(targ_timeStamp > wound_timeStamp - timeEps_ &&
         targ_timeStamp < wound_timeStamp + timeEps_)
      {
        retIndex = i;
	break;
      } // end if(targ_timeStamp == wound_timeStamp)
      else
      {
        // debugging...
        // cerr << " target = " << targ_timeStamp << " <= " << wound_timeStamp;
        // cerr << " - " << timeEps_ << " || " << targ_timeStamp << " >= ";
        // cerr << wound_timeStamp << " + " << timeEps_ << endl;
      }
    } // end if(injured_tissue_ && injured_tissue_->size > 0)
  } // end for(int i = 0; i < injured_tissue_list_.size(); i++)
  if(retIndex == -1) // restore previous injury list
    injured_tissue_ = saveInjury;
  return retIndex;
} // end HotBox::get_timeStep()

void
HotBox::execInjuryList()
{
  // only execute if time has changed
  if(lastTime_ >= currentTime_ - timeEps_ &&
         currentTime_ + timeEps_ >= lastTime_)
     return;

  char message[256];

  // get the injury list for the current time
  int currentTime_step = get_timeStep(currentTime_);

  // report number of injuries read
  cerr << "HotBox::execInjuryList(): timeStep[" << currentTime_step << "] ";
  cerr << injured_tissue_->size() << " injuries found" << endl;

  if(is_diagnosis(*injured_tissue_))
     gui_diagnosis_.set(get_diagnosis(*injured_tissue_));

} // end execInjuryList()

void
HotBox::makeInjGeometry()
{
  int lvindx = 0, numLines = 0;
  int mvindx = 0, numQuads = 0;
  CurveMesh *cm = (CurveMesh *)0;
  QuadSurfMesh *qsm = (QuadSurfMesh *)0;
  vector<double> injIconData;

  CurveField<double> *cf;
  QuadSurfField<double> *qsf;

  cerr << "HotBox::makeInjGeometry(): ";
  cerr << injured_tissue_->size() << " injuries ";
  // traverse the injured tissue list
  for(unsigned int i = 0; i < injured_tissue_->size(); i++)
  {
    VH_injury injPtr = (VH_injury)(*injured_tissue_)[i];

    if(injPtr.geom_type == "line")
    {
      // if it does not already exist, make the CurveMesh
      if(cm == (CurveMesh *)0)
        cm = new CurveMesh();

      cm->add_point(Point(injPtr.axisX0, injPtr.axisY0, injPtr.axisZ0));
      cm->add_point(Point(injPtr.axisX1, injPtr.axisY1, injPtr.axisZ1));
      cm->add_edge(lvindx, lvindx+1);
      lvindx += 2;
      numLines++;
    }
    else if(injPtr.geom_type == "sphere" ||
            injPtr.geom_type == "hollow_sphere")
    {
      // if it does not already exist, make the QuadSurfMesh
      if(qsm == (QuadSurfMesh *)0)
        qsm = new QuadSurfMesh();

      int svindx = 0;
      // make the polygons in the surface of the sphere
      for(int k = 0; k <= CYLREZ; k++)
      { // for each longitudinal step
        for(int j = 0; j <= CYLREZ/2; j++)
        { // make a circle in the X-Y plane
          double pi = 3.14159;
          double
          x = injPtr.rad0 * sin(2.0 * pi * j/CYLREZ) * cos(2.0 * pi * k/CYLREZ);
          double
          y = injPtr.rad0 * cos(2.0 * pi * j/CYLREZ);
          double
          z = injPtr.rad0 * sin(2.0 * pi * j/CYLREZ) * sin(2.0 * pi * k/CYLREZ);
          // rotate the circle into the plane defined by the longitudinal axis
	  Point p(x, y, z, 1.0);
          // translate to the center of the sphere
          p += Vector(injPtr.axisX0, injPtr.axisY0, injPtr.axisZ0);

          qsm->add_point(p);
          // add a data value per mesh node
          if(injPtr.isAblate)
              injIconData.push_back(1.0);
          else if(injPtr.isStun)
              injIconData.push_back(0.5);
          else
              injIconData.push_back(0.0);
          if(svindx >= CYLREZ/2)
          {
            qsm->add_quad(mvindx-(CYLREZ/2)-2, mvindx-(CYLREZ/2)-1,
                          mvindx-(CYLREZ/2)+1, mvindx-(CYLREZ/2));
            numQuads++;
          }
          mvindx += 2; svindx += 2;
        } // end for(int j = 1; j <= CYLREZ; j++)
      } // end for(int k = 1; k <= CYLREZ; k++)
    } // end else if(injPtr.geom_type == "sphere" ... )
    else if(injPtr.geom_type == "cylinder" ||
            injPtr.geom_type == "hollow_cylinder")
    {
      // get the axis of the cylinder
      Vector cylAxis = Point(injPtr.axisX1, injPtr.axisY1, injPtr.axisZ1) -
                       Point(injPtr.axisX0, injPtr.axisY0, injPtr.axisZ0);
      Vector zAxis = cylAxis;
      zAxis.safe_normalize();

      // build the matrix which transforms the cylinder ends
      // into the planes defined by the axis

      Transform cylXform;
      cylXform.rotate(Vector(0.0, 0.0, 1.0), zAxis);

      // make the QuadSurfMesh
      if(qsm == (QuadSurfMesh *)0)
        qsm = new QuadSurfMesh();

      int cvindx = 0;
      // make the polygons in the surface of the cylinder
      for(int j = 0; j <= CYLREZ; j++)
      { // make a circle in the X-Y plane
        double pi = 3.14159;
        double x = injPtr.rad0 * cos(2.0 * pi * j/CYLREZ);
        double y = injPtr.rad0 * sin(2.0 * pi * j/CYLREZ);
        // rotate the circle into the plane defined by the cylindrical axis
        Point p0 = cylXform.project(Point(x, y, 0.0, 1.0));
        p0 += Vector(injPtr.axisX0, injPtr.axisY0, injPtr.axisZ0);
        qsm->add_point(p0);
        x = injPtr.rad1 * cos(2.0 * pi * j/CYLREZ);
        y = injPtr.rad1 * sin(2.0 * pi * j/CYLREZ);
        Point p1 = cylXform.project(Point(x, y, 0.0, 1.0));
        p1 += Vector(injPtr.axisX1, injPtr.axisY1, injPtr.axisZ1);
        qsm->add_point(p1);
        // add a data value per mesh node
        if(injPtr.isAblate)
        {
            injIconData.push_back(1.0);
            injIconData.push_back(1.0);
        }
        else if(injPtr.isStun)
        {
            injIconData.push_back(0.5);
            injIconData.push_back(0.5);
        }
        else
        {
            injIconData.push_back(0.0);
            injIconData.push_back(0.0);
        }
        if(cvindx > 1)
        {
          qsm->add_quad(mvindx-2, mvindx-1, mvindx+1, mvindx);
          numQuads++;
        }
        mvindx += 2; cvindx += 2;
      } // end for(int j = 1; j <= CYLREZ; j++)
    } // end else if(injPtr.geom_type == "cylinder" ... )
  } // end for(int i = 0; i < injured_tissue_->size(); i++)
  if(numLines > 0)
  {
    cerr << numLines << " lines " << injIconData.size() << " data vals ";
    cf = scinew CurveField<double>(cm, -1);
  }
  if(numQuads > 0)
  {
    cerr << numQuads << " quads " << injIconData.size() << " data vals ";
    qsf = scinew QuadSurfField<double>(qsm, 1);
    vector<double>::iterator dptr = injIconData.begin();
    vector<double>::iterator dend = injIconData.end();
    SCIRun::QuadSurfMesh::Node::index_type index = 0;
    for(; dptr != dend; dptr++)
    {
       qsf->set_value(*dptr, index);
       index = index + 1;
    }
    injIconFieldHandle_ = qsf;
    injIconData.clear();
  }

  cerr << " done" << endl;
} // end makeInjGeometry()

/*****************************************************************************
 * method HotBox::executePhysio()
 *****************************************************************************/

void
HotBox::executePhysio()
{
  // skip execution if the current selection and time have not changed
  if(currentSelection_ == lastSelection_)
     return;

  // get the name of the Nrrd file containing the parameters for this selection
  char *nrrdFileName = 
                 hipVarFileList_->get_HIPvarFile((char *)
                                                currentSelection_.c_str());

  if(!nrrdFileName)
  { // there are no physiological parameters corresponding to the selection
    return;
  }
  // pre-pend full directory path to file name
  const string hipVarPath(hipvarpath_.get());
  string nrrdFileNameStr(nrrdFileName);
  string nrrdPathNameStr = hipVarPath + "/" + nrrdFileNameStr;

  // Read the status of this file so we can compare modification timestamps.
  struct stat statbuf;
  if (stat(nrrdPathNameStr.c_str(), &statbuf) == - 1)
  {
    error(string("NrrdReader error - file not found: '")+nrrdPathNameStr+"'");
    return;
  }

  // (else) read the Nrrd file
  InputNrrdHandle_ = 0;
  int namelen = nrrdPathNameStr.size();
  const string ext(".nd");

  // check that the last 3 chars are .nd for us to pio
  if (nrrdPathNameStr.substr(namelen - 3, 3) == ext)
  {
    Piostream *stream = auto_istream(nrrdPathNameStr);
    if (!stream)
    {
      error("Error reading file '" + nrrdPathNameStr + "'.");
      return;
    }

    // Read the file
    Pio(*stream, InputNrrdHandle_);
    if (!InputNrrdHandle_.get_rep() || stream->error())
    {
      error("Error reading data from file '" + nrrdPathNameStr +"'.");
      delete stream;
      return;
    }
    delete stream;
  }
  else
  { // assume it is just a nrrd
    // ICU Monitor needs Properties section of NrrdData file
    error("Input file '" + nrrdPathNameStr +"' must be a '.nd' file");
  }
} // end executePhysio()

/*****************************************************************************
 * method HotBox::executeHighlight()
 *****************************************************************************/
void
HotBox::executeHighlight()
{
  struct stat buf;

  // skip execution if the currentSelection has not changed
  if(currentSelection_ == lastSelection_ &&
         lastTime_ >= currentTime_ - timeEps_ &&
         currentTime_ + timeEps_ >= lastTime_)
     return;

  const string geometryPath(geometrypath_.get());
  if( geometryPath == "" )
  {
    error("Path to segmentation geometry is unset -- please set the directory");
    return;
  }
  // set the file name from the path and current selection
  char filePrefix[256];
  space_to_underbar(filePrefix, (char *)currentSelection_.c_str());
  string selectGeomFilename = geometryPath;
  selectGeomFilename += "/";
  selectGeomFilename += filePrefix;
  selectGeomFilename += ".fld";

  cerr << "executeHighlight: selection " << selectGeomFilename << endl;

//  if (stat(selectGeomFilename.c_str(), &buf)) {
//    remark("File '" + selectGeomFilename + "' not found.");
//    selectGeomFilename = geometryPath;
//    selectGeomFilename += "/";
//    selectGeomFilename += filePrefix;
//    selectGeomFilename += ".fld";

    if (stat(selectGeomFilename.c_str(), &buf)) {
      error("File '" + selectGeomFilename + "' not found.");
      return;
    }
//  }

  Piostream *selectstream = auto_istream(selectGeomFilename);
  if (!selectstream)
  {
    error("Error reading file '" + selectGeomFilename + "'.");
    return;
  }

  // Read the selected geometry highlight file
  Pio(*selectstream, selectGeomFilehandle_);
  if (!selectGeomFilehandle_.get_rep() || selectstream->error())
  {
    error("Error reading data from file '" + selectGeomFilename +"'.");
    delete selectstream;
    return;
  }
  delete selectstream;

  // set injury colors
  // for the first two elements of the injury list
  if(injured_tissue_->size() > 0)
  {
    VH_injury injPtr = (VH_injury)(*injured_tissue_)[0];
    space_to_underbar(filePrefix, (char *)injPtr.anatomyname.c_str());
    string inj0GeomFilename = geometryPath;
    inj0GeomFilename += "/";
    inj0GeomFilename += filePrefix;
    inj0GeomFilename += ".fld";

    cerr << "executeHighlight: injury 0 " << inj0GeomFilename << endl;

    if (stat(inj0GeomFilename.c_str(), &buf)) {
      error("File '" + inj0GeomFilename + "' not found.");
      return;
    }

    Piostream *inj0stream = auto_istream(inj0GeomFilename);
    if (!inj0stream)
    {
      error("Error reading file '" + inj0GeomFilename + "'.");
      return;
    }

    // Read the injury 0 highlight geometry file
    Pio(*inj0stream, inj0GeomFilehandle_);
    if (!inj0GeomFilehandle_.get_rep() || inj0stream->error())
    {
      error("Error reading data from file '" + inj0GeomFilename +"'.");
      delete inj0stream;
      return;
    }
    if(inj0GeomFilehandle_->get_type_description(0)->get_name() !=
	"TriSurfField")
    {
      cerr << "Error -- input field isn't a TriSurfField (typename=";
      cerr << inj0GeomFilehandle_->get_type_description(0)->get_name();
    }
    else
    {
      // get the mesh from the input TriSurfField
      TriSurfMeshHandle
      inj0GeomMeshH = dynamic_cast<TriSurfMesh*>
          (inj0GeomFilehandle_->mesh().get_rep());
      // iterate over the mesh nodes to create the corresponding data
      TriSurfMesh::Node::iterator iter;
      TriSurfMesh::Node::iterator eiter;
      inj0GeomMeshH->begin(iter);
      inj0GeomMeshH->end(eiter);
      // make the output TriSurfField
      TriSurfField<double>
      *tsf = scinew TriSurfField<double>(inj0GeomMeshH, 1);
      vector<double> injIconData;
      injIconData.push_back((double)injPtr.probability);
      vector<double>::iterator dataIter = injIconData.begin();
      // set the data in the output TriSurfField
      TriSurfMesh::Node::index_type index = 0;
      for(; iter != eiter; ++iter)
      {
         tsf->set_value(*dataIter, index);
         index = index + 1;
      }
      inj0GeomFieldhandle_ = tsf;
    }
    delete inj0stream;
  } // end if(injured_tissue_->size() > 0)
  if(injured_tissue_->size() > 1)
  {
    VH_injury injPtr = (VH_injury)(*injured_tissue_)[1];
    space_to_underbar(filePrefix, (char *)injPtr.anatomyname.c_str());
    string inj1GeomFilename = geometryPath;
    inj1GeomFilename += "/";
    inj1GeomFilename += filePrefix;
    inj1GeomFilename += ".fld";

    cerr << "executeHighlight: injury 1 " << inj1GeomFilename << endl;

    if (stat(inj1GeomFilename.c_str(), &buf)) {
      error("File '" + inj1GeomFilename + "' not found.");
      return;
    }

    Piostream *inj1stream = auto_istream(inj1GeomFilename);
    if (!inj1stream)
    {
      error("Error reading file '" + inj1GeomFilename + "'.");
      return;
    }

    // Read the file
    Pio(*inj1stream, inj1GeomFilehandle_);
    if (!inj1GeomFilehandle_.get_rep() || inj1stream->error())
    {
      error("Error reading data from file '" + inj1GeomFilename +"'.");
      delete inj1stream;
      return;
    }
    if(inj1GeomFilehandle_->get_type_description(0)->get_name() !=
	"TriSurfField")
    {
      cerr << "Error -- input field isn't a TriSurfField (typename=";
      cerr << inj1GeomFilehandle_->get_type_description(0)->get_name();
    }
    else
    {
      // get the mesh from the input TriSurfField
      TriSurfMeshHandle
      inj1GeomMeshH = dynamic_cast<TriSurfMesh*>
          (inj1GeomFilehandle_->mesh().get_rep());
      // iterate over the mesh nodes to create the corresponding data
      TriSurfMesh::Node::iterator iter;
      TriSurfMesh::Node::iterator eiter;
      inj1GeomMeshH->begin(iter);
      inj1GeomMeshH->end(eiter);
      // make the output TriSurfField
      TriSurfField<double>
      *tsf = scinew TriSurfField<double>(inj1GeomMeshH, 1);
      // set the data in the output TriSurfField
      vector<double> injIconData;
      injIconData.push_back((double)injPtr.probability);
      vector<double>::iterator dataIter = injIconData.begin();
      TriSurfMesh::Node::index_type index = 0;
      for(; iter != eiter; ++iter)
      {
         tsf->set_value(*dataIter, index);
         index = index + 1;
      }
      inj1GeomFieldhandle_ = tsf;
    }
    delete inj1stream;
  } // end if(injured_tissue_->size() > 1)
} // end executeHighlight()

void
HotBox::widget_moved(bool last, BaseWidget*)
{
  if (last)
  {
    want_to_execute();
  }
}

void
HotBox::executeProbe() {

  const string selectionSource(selectionsource_.get());
  if(selectionSource == "fromProbe")
  { // get current position of widget
    probeLoc_ = probeWidget_->GetPosition();
    cerr << "HotBox::executeProbe(): probeWidget->Position: ";
    cerr << probeLoc_ << endl;
    // update the probe location in the Tcl GUI
    gui_probeLocx_.set(probeLoc_.x());
    gui_probeLocy_.set(probeLoc_.y());
    gui_probeLocz_.set(probeLoc_.z());
  }

  typedef LatVolField<unsigned short> FLD;

  FLD *labels = dynamic_cast<FLD*>(InputFieldHandle_.get_rep());

  if (! labels) {
    error("HotBox::executeProbe(): Expected LatVolField<unsigned short>!");
    return;
  }
  LatVolMesh::Node::index_type index;
  LatVolMeshHandle lvmh = labels->get_typed_mesh();
  if (!lvmh->locate(index, probeLoc_)) {
    warning("Probe is outside of label volume.");
    labelIndexVal_ = 0;
    return;
  }
  unsigned short val;
  labels->value(val, index);
  labelIndexVal_ = val;
} // end HotBox::executeProbe()

void
 HotBox::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace VS


