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
#include <Core/Datatypes/ColumnMatrix.h>
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
#include <Dataflow/XMLUtil/XMLUtil.h>
#include <Dataflow/XMLUtil/StrX.h>

#include <Dataflow/share/share.h>

#include <sys/stat.h>
#include <string.h>
#include <iostream>
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

  // toggle on/off drawing GeomSticky output
  GuiString enableDraw_;

  // the HotBox interaction
  VS_SCI_Hotbox *VS_HotBoxUI;

  // file or OQAFMA
  GuiInt datasource_;

  // Query Type
  GuiInt querytype_;

  // "fromHotBoxUI" or "fromProbe"
  GuiString selectionsource_;

  GuiString anatomydatasource_;
  GuiString adjacencydatasource_;
  GuiString boundingboxdatasource_;
  GuiString injurylistdatasource_;
  GuiString currentselection_;

  // temporary:  fixed anatomical label map files
  VH_MasterAnatomy *anatomytable;
  VH_AdjacencyMapping *adjacencytable;
  VH_AnatomyBoundingBox *boundBoxList;

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
  gui_is_injured1_(ctx->subVar("gui_is_injured1")),
  gui_is_injured2_(ctx->subVar("gui_is_injured2")),
  gui_is_injured3_(ctx->subVar("gui_is_injured3")),
  gui_is_injured4_(ctx->subVar("gui_is_injured4")),
  gui_is_injured5_(ctx->subVar("gui_is_injured5")),
  gui_is_injured6_(ctx->subVar("gui_is_injured6")),
  gui_is_injured7_(ctx->subVar("gui_is_injured7")),
  gui_is_injured8_(ctx->subVar("gui_is_injured8")),
  gui_is_injured9_(ctx->subVar("gui_is_injured9")),
  enableDraw_(ctx->subVar("enableDraw")),
  datasource_(ctx->subVar("datasource")),
  querytype_(ctx->subVar("querytype")),
  selectionsource_(ctx->subVar("selectionsource")),
  anatomydatasource_(ctx->subVar("anatomydatasource")),
  adjacencydatasource_(ctx->subVar("adjacencydatasource")),
  boundingboxdatasource_(ctx->subVar("boundingboxdatasource")),
  injurylistdatasource_(ctx->subVar("injurylistdatasource")),
  currentselection_(ctx->subVar("currentselection"))
{
  // instantiate the HotBox-specific interaction structure
  VS_HotBoxUI = new VS_SCI_Hotbox();
  // Start the Java Virtual Machine
  // Initialize our interface (JNI via JACE)
  // to the Protege Foundational Model of Anatomy (FMA)
  // temporary -- use fixed text files
  anatomytable = new VH_MasterAnatomy();
  adjacencytable = new VH_AdjacencyMapping();
  boundBoxList = (VH_AnatomyBoundingBox *)NULL;
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

  } // end if(InputFieldHandle->query_scalar_interface()...)
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

  const int dataSource(datasource_.get());
  const int queryType(querytype_.get());
  const string selectionSource(selectionsource_.get());
  const string currentSelection(currentselection_.get());
  const string anatomyDataSrc(anatomydatasource_.get());
  const string adjacencyDataSrc(adjacencydatasource_.get());
  const string boundingBoxDataSrc(boundingboxdatasource_.get());
  const string injuryListDataSrc(injurylistdatasource_.get());
  const string enableDraw(enableDraw_.get());

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
  if(!anatomytable->get_num_names())
  { // label maps have not been read
    anatomytable->readFile((char *)anatomyDataSrc.c_str());
  }
  else
  {
    cout << "Master Anatomy file contains " << anatomytable->get_num_names();
    cout << " names" << endl;
  }

  // if the selection source is from the HotBox UI -- ignore the probe
  char partsName[256], capitalName[256], selectName[256];

  if(selectionSource == "fromHotBoxUI")
  {
    strcpy(selectName, currentSelection.c_str());
    // clear selection source
    selectionsource_.set("fromProbe");
  }
  else if(anatomytable->get_anatomyname(labelIndexVal) != 0)
    strcpy(selectName, anatomytable->get_anatomyname(labelIndexVal));
  else
    strcpy(selectName, "");

  if(strlen(selectName) != 0)
    cout << "VS/HotBox: selected '" << selectName << "'" << endl;
  else
    remark("Selected [NULL]");

  if( boundingBoxDataSrc == "" ) {
    error("No Bounding Box file has been selected.  Please choose a file.");
    return;
  }
  if(!boundBoxList)
  { // bounding boxes have not been read
    if (stat(boundingBoxDataSrc.c_str(), &buf)) {
    error("File '" + boundingBoxDataSrc + "' not found.");
    return;
    }

    boundBoxList =
         VH_Anatomy_readBoundingBox_File((char *)boundingBoxDataSrc.c_str());
  }

  // get the bounding box information for the selected entity
  VH_AnatomyBoundingBox *selectBox =
      VH_Anatomy_findBoundingBox( boundBoxList, selectName);

  // Read the Injury List -- Every time the HotBox Evaluates
  try {
    XMLPlatformUtils::Initialize();
  } catch (const XMLException& toCatch) {
    std::cerr << "Error during initialization! :\n"
         << StrX(toCatch.getMessage()) << endl;
    return;
  }

  // Instantiate a DOM parser for the injury list file.
  XercesDOMParser injListParser;
  injListParser.setDoValidation(false);

  try {
    injListParser.parse(injuryListDataSrc.c_str());
  }  catch (const XMLException& toCatch) {
    std::cerr << "Error during parsing: '" <<
      injuryListDataSrc << "'\nException message is:  " <<
      xmlto_string(toCatch.getMessage());
      return;
  }
  // we are interested in injured tissues -- look for "region"
  // <wound entity="..." timestamp="...1">
  //        <primaryInjury>
  //            <ablate/stunRegion probability="...">
  //                <region entity="...tissue name..."/> 
  //            </ablateRegion>
  //        </primaryInjury>
  // </wound>


  DOMDocument *injListDoc = injListParser.getDocument();
  DOMNodeList *
  injList = injListDoc->getElementsByTagName(to_xml_ch_ptr("region"));
  unsigned long i, num_struQLret, num_injList = injList->getLength();
  char *injured_tissue[VH_LM_NUM_NAMES];

  if (num_injList == 0) {
    cout << "HotBox.cc: no entities in Injury List" << endl;
  }
  else
  {
    cout << "HotBox.cc: xml file: " << injuryListDataSrc << ": "
         << num_injList << " wounded region entities" << endl;
    if(num_injList >= VH_LM_NUM_NAMES)
           num_injList = VH_LM_NUM_NAMES;
    for (i = 0;i < num_injList; i++)
    {
      if (!(injList->item(i)))
      {
        std::cerr << "Error: NULL DOM node" << std::endl;
        continue;
      }
      DOMNode &node = *(injList->item(i));
      cout << "Node[" << i << "] type: " << node.getNodeType();
      cout << " name: " << to_char_ptr(node.getNodeName());
      if(node.hasAttributes())
      {
          cout << " " << node.getAttributes()->getLength()
               << " attributes" << endl;
          DOMNode *
          elem = node.getAttributes()->item(0);
          if(elem == 0)
              cout << " Cannot get element from node" << endl;
          else
          {
              cout << " value: " << to_char_ptr(elem->getNodeValue()) << endl;
              injured_tissue[i] = strdup(to_char_ptr(elem->getNodeValue()));
          }
      } // end if(node.hasAttributes())
    } // end for (i = 0;i < num_injList; i++)
  } // end else (num_injList != 0)
  // we now have the anatomy name corresponding to the label value at the voxel
  char *oqafma_relation[VH_LM_NUM_NAMES];
  if(dataSource == VS_DATASOURCE_OQAFMA)
  {
    fprintf(stderr, "dataSource = OQAFMA\n");
    ns1__processStruQLResponse resultStruQL;
    ServiceInterfaceSoapBinding ws;
    // build the query to send to OQAFMA
    std::string p2;
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
        for (i = 0;i < num_struQLret; i++)
        {
          if (!(struQLretList->item(i)))
          {
            cout << "Error: NULL DOM node" << std::endl;
            continue;
          }
          DOMNode &node = *(struQLretList->item(i));
          cout << "Node[" << i << "] type: " << node.getNodeType();
          cout << " name: " << to_char_ptr(node.getNodeName());
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

  } // end if(dataSource == VS_DATASOURCE_OQAFMA)
  else // dataSource == FILES
  {
    fprintf(stderr, "dataSource = FILES[%d]\n", dataSource);
    // use fixed Adjacency Map files
    if(!adjacencytable->get_num_names())
    { // adjacency data has not been read
      adjacencytable->readFile((char *)adjacencyDataSrc.c_str());
    }
    else
    {
      cout << "Adjacency Map file contains " << adjacencytable->get_num_names();
      cout << " entries" << endl;
    }
  } // end else(use fixed Adjacency Map files)

  // draw HotBox Widget
  GeomGroup *HB_geomGroup = scinew GeomGroup();
  Color text_color;
  text_color = Color(1,1,1);
  MaterialHandle text_material = scinew Material(text_color);
  text_material->transparency = 0.75;

  GeomLines *lines = scinew GeomLines();
  GeomTexts *texts = scinew GeomTexts();

  VS_HotBoxUI->setOutput(lines, texts);
  VS_HotBoxUI->setOutMtl(text_material);

  // get the adjacency info for the selected entity
  int *adjPtr = adjacencytable->adjacent_to(labelIndexVal);

  // fill in text labels in the HotBox
  char *adjacentName;
  if(adjacencytable->get_num_rel(labelIndexVal) >= 1)
  { // Note: TCL UI is 1-indexed, row-major
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
    if(is_injured(adjacentName, injured_tissue, num_injList))
    { gui_is_injured1_.set(1); }
    else
    { gui_is_injured1_.set(0); }
    // OpenGL UI is indexed 0-7, column-major
    VS_HotBoxUI->set_text(0, string(adjacentName, 0, 18));
  } // end if(adjacencytable->get_num_rel(labelIndexVal) >= 1)
  if(adjacencytable->get_num_rel(labelIndexVal) >= 2)
  { // Note: TCL UI is 1-indexed, row-major
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
    if(is_injured(adjacentName, injured_tissue, num_injList))
    { gui_is_injured2_.set(1); }
    else
    { gui_is_injured2_.set(0); }
    // OpenGL UI is indexed 0-7, column-major
    VS_HotBoxUI->set_text(3, string(adjacentName, 0, 18));
  } // end if(adjacencytable->get_num_rel(labelIndexVal) >= 2)
  if(adjacencytable->get_num_rel(labelIndexVal) >= 3)
  {  // Note: TCL UI is 1-indexed, row-major
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
    if(is_injured(adjacentName, injured_tissue, num_injList))
    { gui_is_injured3_.set(1); }
    else
    { gui_is_injured3_.set(0); }
    // OpenGL UI is indexed 0-7, column-major
    VS_HotBoxUI->set_text(5, string(adjacentName, 0, 18));
  } // end if(adjacencytable->get_num_rel(labelIndexVal) >= 3)
  if(adjacencytable->get_num_rel(labelIndexVal) >= 4)
  { // Note: TCL UI is 1-indexed, row-major
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
    if(is_injured(adjacentName, injured_tissue, num_injList))
    { gui_is_injured4_.set(1); }
    else
    { gui_is_injured4_.set(0); }
    // OpenGL UI is indexed 0-7, column-major
    VS_HotBoxUI->set_text(1, string(adjacentName, 0, 18));
  } // end if(adjacencytable->get_num_rel(labelIndexVal) >= 4)

  gui_label5_.set(selectName);
  VS_HotBoxUI->set_text(5, string(selectName, 0, 18));
  currentselection_.set(selectName);
  
  if(adjacencytable->get_num_rel(labelIndexVal) >= 6)
  { // Note: TCL UI is 1-indexed, row-major
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
    if(is_injured(adjacentName, injured_tissue, num_injList))
    { gui_is_injured6_.set(1); }
    else
    { gui_is_injured6_.set(0); }
    // OpenGL UI is indexed 0-7, column-major
    VS_HotBoxUI->set_text(6, string(adjacentName, 0, 18));
  } // end if(adjacencytable->get_num_rel(labelIndexVal) >= 6)
  if(adjacencytable->get_num_rel(labelIndexVal) >= 7)
  { // Note: TCL UI is 1-indexed, row-major
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
    if(is_injured(adjacentName, injured_tissue, num_injList))
    { gui_is_injured7_.set(1); }
    else
    { gui_is_injured7_.set(0); }
    // OpenGL UI is 0-indexed, column-major
    VS_HotBoxUI->set_text(2, string(adjacentName, 0, 18));
  } // end if(adjacencytable->get_num_rel(labelIndexVal) >= 7)
  if(adjacencytable->get_num_rel(labelIndexVal) >= 8)
  { // Note: TCL UI is 1-indexed, row-major 
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
    if(is_injured(adjacentName, injured_tissue, num_injList))
    { gui_is_injured8_.set(1); }
    else
    { gui_is_injured8_.set(0); }
    // OpenGL UI is indexed 0-7, column-major
    VS_HotBoxUI->set_text(4, string(adjacentName, 0, 18));
  } // end if(adjacencytable->get_num_rel(labelIndexVal) >= 8)
  if(adjacencytable->get_num_rel(labelIndexVal) >= 9)
  { // Note: TCL UI is 1-indexed, row-major
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
    if(is_injured(adjacentName, injured_tissue, num_injList))
    { gui_is_injured9_.set(1); }
    else
    { gui_is_injured9_.set(0); }
    // OpenGL UI is indexed 0-7, column-major
    VS_HotBoxUI->set_text(7, string(adjacentName, 0, 18));
  } // end if(adjacencytable->get_num_rel(labelIndexVal) >= 9)

  // clean up
  if(dataSource == VS_DATASOURCE_OQAFMA)
  {
     for (i = 0;i < num_struQLret; i++)
     {
       if(oqafma_relation[i] != 0)
       {
         free(oqafma_relation[i]);
         oqafma_relation[i] = 0;
       }
     }
     num_struQLret = 0;
  } // end if(dataSource == VS_DATASOURCE_OQAFMA)

  for(i = 0;i < num_injList; i++)
  {
    if(injured_tissue[i] != 0)
    {
       free(injured_tissue[i]);
       injured_tissue[i] = 0;
    }
  }
  num_injList = 0;

  if(enableDraw == "yes")
  {
    VS_HotBoxUI->draw(0, 0, 0.005);
  
    HB_geomGroup->add(lines);
    HB_geomGroup->add(texts);
  }

  // set output geometry port -- hotbox drawn to viewer
  GeometryOPort *outGeomPort = (GeometryOPort *)get_oport("HotBox Widget");
  if(!outGeomPort) {
    error("Unable to initialize output geometry port.");
    return;
  }
  GeomSticky *sticky = scinew GeomSticky(HB_geomGroup);
  outGeomPort->delAll();
  outGeomPort->addObj( sticky, "HotBox Sticky" );
  outGeomPort->flushViews();

  // set output matrix port -- bounding box of selection
  MatrixOPort *outMatrixPort = (MatrixOPort *)get_oport("Bounding Box");
  if(!outMatrixPort) {
    error("Unable to initialize output matrix port.");
    return;
  }

  MatrixHandle cm = scinew ColumnMatrix(6);
  if(selectBox)
  {
    cm->get(0, 0) = (double)selectBox->minX;
    cm->get(1, 0) = (double)selectBox->minY;
    cm->get(2, 0) = (double)selectBox->minZ;
    cm->get(3, 0) = (double)selectBox->maxX;
    cm->get(4, 0) = (double)selectBox->maxY;
    cm->get(5, 0) = (double)selectBox->maxZ;
  }
  outMatrixPort->send(cm);


} // end HotBox::execute()

void
 HotBox::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace VS


