/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

/* ComponentNode.cc
 * 
 * written by 
 *   Chris Moulding
 *   Sept 2000
 *   Copyright (c) 2000
 *   University of Utah
 */

#include <Core/Util/RWS.h>
#include <Dataflow/Network/ComponentNode.h>
#include <Dataflow/XMLUtil/XMLUtil.h>
#include <Dataflow/Network/PackageDBHandler.h>
#include <Dataflow/Network/StrX.h>

#include <iostream>
#include <fstream>
#include <strstream>
#include <stdlib.h>

#ifdef __sgi
#define IRIX
#pragma set woff 1375
#endif
#include <util/PlatformUtils.hpp>
#include <sax/SAXException.hpp>
#include <sax/SAXParseException.hpp>
#include <parsers/DOMParser.hpp>
#include <dom/DOM_NamedNodeMap.hpp>
#include <sax/ErrorHandler.hpp>
#ifdef __sgi
#pragma reset woff 1375
#endif

using std::map;
using std::cout;
using std::endl;
using std::strstream;


#define rWSgSC(x) removeLTWhiteSpace(getSerializedChildren(x))

namespace SCIRun {

typedef map<int,char*>::iterator char_iter;
typedef map<int,parameter_node*>::iterator param_iter;
typedef map<int,plan_node*>::iterator plan_iter;
typedef map<int,inport_node*>::iterator inport_iter;
typedef map<int,outport_node*>::iterator outport_iter;
typedef map<int,file_node*>::iterator file_iter;
typedef map<int,device_node*>::iterator device_iter;
typedef map<int,implementation_node*>::iterator impl_iter;

component_node* CreateComponentNode(int type)
{
  component_node* n = new component_node;
  switch (type) {
  case 1: /* everything */
    n->name = NOT_SET;
    n->category = NOT_SET;
    n->overview = new overview_node;
    n->overview->authors = new map<int,char*>;
    n->overview->summary = NOT_SET;
    n->overview->description = NOT_SET;
    n->overview->examplesr = NOT_SET;
    n->implementation = new implementation_node;
    n->implementation->ccfiles = new map<int,char*>;
    n->implementation->cfiles = new map<int,char*>;
    n->implementation->ffiles = new map<int,char*>;
    n->io = new io_node;
    n->io->lastportdynamic = 1;
    n->io->infiles = new map<int,file_node*>;
    n->io->outfiles = new map<int,file_node*>;
    n->io->indevices = new map<int,device_node*>;
    n->io->outdevices = new map<int,device_node*>;
    n->io->inports = new map<int,inport_node*>;
    n->io->outports = new map<int,outport_node*>;
    n->gui = new gui_node;
    n->gui->description = NOT_SET;
    n->gui->image = NOT_SET;
    n->gui->parameters = new map<int,parameter_node*>;
    n->testingplans = new map<int,plan_node*>;
    break;
  case 2: /* name and category only */
    n->name = NOT_SET;
    n->category = NOT_SET;
    n->overview = NULL;
    n->implementation = NULL;
    n->io = NULL;
    n->gui = NULL;
    n->testingplans = NULL;
    break;
  case 3: /* name, category and ports */
    n->name = NOT_SET;
    n->category = NOT_SET;
    n->overview = NULL;
    n->implementation = NULL;
    n->io = new io_node;
    n->io->infiles = NULL;
    n->io->outfiles = NULL;
    n->io->indevices = NULL;
    n->io->outdevices = NULL;
    n->io->inports = new map<int,inport_node*>;
    n->io->outports = new map<int,outport_node*>;
    n->io->lastportdynamic = 1;
    n->gui = NULL;
    n->testingplans = NULL;
    break;
  }

  return n;
}

void DestroyOverviewNode(overview_node* n)
{
  if (n->authors) {
    for(char_iter i=n->authors->begin();
	i!=n->authors->end();
	i++) 
      delete[] (*i).second; 
    delete n->authors;
  }
  if (n->summary && n->summary!=NOT_SET) delete[] n->summary;
  if (n->description && n->description!=NOT_SET) delete[] n->description;
  delete n;
}

void DestroyFileNode(file_node* n)
{
  if (n->description && n->description!=NOT_SET) delete[] n->description;
  if (n->datatype && n->datatype!=NOT_SET) delete[] n->datatype;
  delete n;
}

void DestroyInportNode(inport_node* n)
{
  if (n->name && n->name!=NOT_SET) delete[] n->name;
  if (n->description && n->description!=NOT_SET) delete[] n->description;
  if (n->datatype && n->datatype!=NOT_SET) delete[] n->datatype;
  if (n->upstream) {
    for (char_iter i=n->upstream->begin();
	 i!=n->upstream->end();
	 i++)
      delete[] (*i).second;
    delete n->upstream;
  }
  delete n;
}

void DestroyOutportNode(outport_node* n)
{
  if (n->name && n->name!=NOT_SET) delete[] n->name;
  if (n->description && n->description!=NOT_SET) delete[] n->description;
  if (n->datatype && n->datatype!=NOT_SET) delete[] n->datatype;
  if (n->downstream) {
    for (char_iter i=n->downstream->begin();
         i!=n->downstream->end();
         i++)
      delete[] (*i).second;
    delete n->downstream;
  }
  delete n;
}

void DestroyDeviceNode(device_node* n)
{
  if (n->devicename && n->devicename!=NOT_SET) delete[] n->devicename;
  if (n->description && n->description!=NOT_SET) delete[] n->description;
}

void DestroyIoNode(io_node* n)
{
  inport_iter i2;
  outport_iter i3;
  file_iter i4;
  device_iter i5;

  if (n->infiles) {
    for(i4=n->infiles->begin();
	i4!=n->infiles->end();
	i4++) 
      DestroyFileNode((*i4).second);
    delete n->infiles;
  }
  if (n->outfiles) {
    for(i4=n->outfiles->begin();
	i4!=n->outfiles->end();
	i4++) 
      DestroyFileNode((*i4).second);
    delete n->outfiles;
  }
  if (n->indevices) {
    for(i5=n->indevices->begin();
	i5!=n->indevices->end();
	i5++) 
      DestroyDeviceNode((*i5).second);
    delete n->indevices;
  }
  if (n->outdevices) {
    for(i5=n->outdevices->begin();
	i5!=n->outdevices->end();
	i5++) 
      DestroyDeviceNode((*i5).second);
    delete n->outdevices;
  }
  if (n->inports) {
    for(i2=n->inports->begin();
	i2!=n->inports->end();
	i2++) 
      DestroyInportNode((*i2).second);
    delete n->inports;
  }
  if (n->outports) {
    for(i3=n->outports->begin();
	i3!=n->outports->end();
	i3++) 
      DestroyOutportNode((*i3).second);
    delete n->outports;
  }
  delete n;
}

void DestroyParameterNode(parameter_node* n)
{
  if (n->widget && n->widget!=NOT_SET) delete[] n->widget;
  if (n->datatype && n->datatype!=NOT_SET) delete[] n->datatype;
  if (n->label && n->label!=NOT_SET) delete[] n->label;
  if (n->description && n->description!=NOT_SET) delete[] n->description;
  delete n;
}

void DestroyGuiNode(gui_node* n)
{
  if (n->description && n->description!=NOT_SET) delete[] n->description;
  if (n->image && n->image!=NOT_SET) delete[] n->image;
  if (n->parameters) {
    for(param_iter i=n->parameters->begin();
	i!=n->parameters->end();
	i++)
      DestroyParameterNode((*i).second);
    delete n->parameters;
  }
  delete n;
}

void DestroyPlanNode(plan_node* n)
{
  if (n->description && n->description!=NOT_SET) delete[] n->description;

  if (n->steps) {
    for (char_iter i=n->steps->begin();
	 i!=n->steps->end();
	 i++)
      delete[] (*i).second;
    delete n->steps;
  }
  delete n;
}

void DestroyImplementationNode(implementation_node* n)
{
  char_iter i;
  for (i=n->ccfiles->begin();
       i!=n->ccfiles->end();
       i++)
    delete[] (*i).second;
  for (i=n->cfiles->begin();
       i!=n->cfiles->end();
       i++)
    delete[] (*i).second;
  for (i=n->ffiles->begin();
       i!=n->ffiles->end();
       i++)
    delete[] (*i).second;
}

void DestroyComponentNode(component_node* n)
{
  if (n->name && n->name!=NOT_SET) delete[] n->name;
  if (n->category && n->category!=NOT_SET) delete[] n->category;
  if (n->overview) DestroyOverviewNode(n->overview);
  if (n->implementation) DestroyImplementationNode(n->implementation);
  if (n->io) DestroyIoNode(n->io);
  if (n->gui) DestroyGuiNode(n->gui);
  if (n->testingplans) {
    for(plan_iter i=n->testingplans->begin();
        i!=n->testingplans->end();
        i++)
      DestroyPlanNode((*i).second);
    delete n->testingplans;
  }
  delete n;
}

void PrintComponentNode(component_node* n)
{
  char_iter i;
  file_iter i2;
  plan_iter i3;
  param_iter i4;
  inport_iter i5;
  outport_iter i6;
  device_iter i7;

  cout << "Component Name: " << n->name << endl << endl;
  cout << "  Category: " << n->category << endl << endl;

  if (n->overview) {
    cout << "  Authors:" << endl;
    if (n->overview->authors)
      for (i=n->overview->authors->begin();
	   i!=n->overview->authors->end();
	   i++)
	cout << "    " << (*i).second << endl;
    cout << endl;
    cout << "  Summary:" << endl;
    cout << "    " << n->overview->summary << endl;
    cout << endl;
    cout << "  Description:" << endl;
    if (n->overview->description)
	cout << "    " << n->overview->description << endl;
    cout << endl;
    cout << "  Example .sr file: " << n->overview->examplesr << endl;
    cout << endl;
  }

  // print the implementation node here

  if (n->io) {
    cout << "  Inputs:" << endl;
    if (n->io->inports->size()) {
      cout << "    Ports:" << endl;
      for (i5=n->io->inports->begin();
	   i5!=n->io->inports->end();
	   i5++) {
	cout << "      name       : " << (*i5).second->name << endl;
	cout << "      Description: " << endl;
	if ((*i5).second->description)
	  cout << "    " << (*i5).second->description << endl;
	cout << "      Datatype   : " << (*i5).second->datatype << endl;
	cout << "      Upstream   : " << endl;
	if ((*i5).second->upstream)
	  for (i=(*i5).second->upstream->begin();
	       i!=(*i5).second->upstream->end();
	       i++)
	    cout << "        " << (*i).second << endl;
	cout << endl;
      }
    }
    if (n->io->infiles->size()) {
      cout << "    Files:" << endl;
      for(i2=n->io->infiles->begin();
	  i2!=n->io->infiles->end();
	  i2++) {
	cout << "      Description: " << endl;
	if ((*i2).second->description)
	  cout << "    " << (*i2).second->description << endl;
	cout << "      Datatype   : " << (*i2).second->datatype << endl;
	cout << endl;
      }
    }
    if (n->io->indevices->size()) {
      cout << "    Devices:" << endl;
      for(i7=n->io->indevices->begin();
	  i7!=n->io->indevices->end();
	  i7++) {
	cout << "      Device Name: " << (*i7).second->devicename << endl;
	cout << "      Description:" << endl;
	  cout << "    " << (*i7).second->description << endl;
	cout << endl;
      }
    }
    cout << "  Outputs:" << endl;
    if (n->io->outports->size()) {
      cout << "    Ports:" << endl;
      for (i6=n->io->outports->begin();
	   i6!=n->io->outports->end();
	   i6++) {
	cout << "      Name       : " << (*i6).second->name << endl;
	cout << "      Description: " << endl;
	cout << "        " << (*i6).second->description << endl;
	cout << "      Datatype   : " << (*i6).second->datatype << endl;
	cout << "      Downstream : " << endl;
	if ((*i6).second->downstream)
	  for (i=(*i6).second->downstream->begin();
	       i!=(*i6).second->downstream->end();
	       i++)
	    cout << "        " << (*i).second << endl;
	cout << endl;
      }
    }
    if (n->io->outfiles->size()) {
      cout << "    Files:" << endl;
      for(i2=n->io->outfiles->begin();
	  i2!=n->io->outfiles->end();
	  i2++) {
	cout << "      Description: " << endl;
	  cout << "        " << (*i2).second->description << endl;
	cout << "      Datatype   : " << (*i2).second->datatype << endl;
	cout << endl;
      }
    }
    if (n->io->outdevices->size()) {
      cout << "    Devices:" << endl;
      for(i7=n->io->outdevices->begin();
	  i7!=n->io->outdevices->end();
	  i7++) {
	cout << "      Device Name: " << (*i7).second->devicename << endl;
	cout << "      Description:" << endl;
	cout << "        " << (*i7).second->description << endl;
	cout << endl;
      }
    }
  }    

  if (n->gui) {
    cout << "  GUI:" << endl;
    cout << "    Description: " << endl;
      cout << "        " << n->gui->description << endl;
    cout << "    Image      : " << n->gui->image << endl;
    cout << "    Parameters : " << endl;
    if (n->gui->parameters)
      for(i4=n->gui->parameters->begin();
	  i4!=n->gui->parameters->end();
	  i4++) {
	cout << "      Label      : " << (*i4).second->label << endl;
	cout << "      Widget     : " << (*i4).second->widget << endl;
	cout << "      Datatype   : " << (*i4).second->datatype << endl;
	cout << "      Description: " << endl;
	  cout << "        " << (*i4).second->description << endl;
	cout << endl;
      }
    cout << endl;
  }
  
  if (n->testingplans) {
    int plancount = 1;
    int stepcount = 1;
    cout << "  Testing plan(s):" << endl;
    for (i3=n->testingplans->begin();
	 i3!=n->testingplans->end();
	 i3++,plancount++) {
      cout << "    Plan " << plancount << ":" << endl; 
      cout << "      Description: " << endl 
	   << (*i3).second->description << endl;
      cout << "      Steps:" << endl;
      for (i=(*i3).second->steps->begin();
	   i!=(*i3).second->steps->end();
	   i++)
	cout << "        Step " << stepcount++ << ":" << endl
	     << "          " << (*i).second << endl;
      cout << endl;
    }
  }
}

void ProcessFileNode(const DOM_Node& d, file_node* n)
{
  for (DOM_Node child = d.getFirstChild();
       child!=0;
       child=child.getNextSibling()) {
    DOMString childname = child.getNodeName();
    if (childname.equals("description") && n->description==NOT_SET)
      n->description = rWSgSC(child);
    else if (childname.equals("datatype") && n->datatype==NOT_SET)
      n->datatype = rWSgSC(child);
  }
}

void ProcessInportNode(const DOM_Node& d, inport_node* n)
{
  for (DOM_Node child = d.getFirstChild();
       child!=0;
       child=child.getNextSibling()) {
    DOMString childname = child.getNodeName();
    if (childname.equals("description") && n->description==NOT_SET)
      n->description = rWSgSC(child);
    else if (childname.equals("name") && n->name==NOT_SET)
      n->name = rWSgSC(child);
    else if (childname.equals("datatype") && n->datatype==NOT_SET)
      n->datatype = rWSgSC(child);
    else if (childname.equals("componentname") && n->upstream)
      n->upstream->insert(std::pair<int,char*>(n->upstream->size(),
					       rWSgSC(child)));
  }				       
}

void ProcessOutportNode(const DOM_Node& d, outport_node* n)
{
  for (DOM_Node child = d.getFirstChild();
       child!=0;
       child=child.getNextSibling()) {
    DOMString childname = child.getNodeName();
    if (childname.equals("description") && n->description==NOT_SET)
      n->description = rWSgSC(child);
    else if (childname.equals("name") && n->name==NOT_SET)
      n->name = rWSgSC(child);
    else if (childname.equals("datatype") && n->datatype==NOT_SET)
      n->datatype = rWSgSC(child);
    else if (childname.equals("componentname") && n->downstream)
      n->downstream->insert(std::pair<int,char*>(n->downstream->size(),
					       rWSgSC(child)));
  }
}

void ProcessDeviceNode(const DOM_Node& d, device_node* n)
{
  for (DOM_Node child = d.getFirstChild();
       child!=0;
       child=child.getNextSibling()) {
    DOMString childname = child.getNodeName();
    if (childname.equals("description") && n->description==NOT_SET)
      n->description = rWSgSC(child);
    else if (childname.equals("devicename") && n->devicename==NOT_SET)
      n->devicename = rWSgSC(child);
  }
}

void ProcessIoNode(const DOM_Node& d, io_node* n)
{
  for (DOM_Node child = d.getFirstChild();
       child!=0;
       child=child.getNextSibling()) {
    DOMString childname = child.getNodeName();
    if (childname.equals("inputs")){
      if (n->lastportdynamic) {
	DOM_Node attr = child.getAttributes().getNamedItem("lastportdynamic");
	if (attr!=0) {
	  DOMString attrstr = attr.getNodeValue();
	  if (attrstr.equals("yes") ||
	      attrstr.equals("y") ||
	      attrstr.equals("YES") ||
	      attrstr.equals("Y"))
	    n->lastportdynamic = 1;
	  else
	    n->lastportdynamic = 0;
	} else
	  n->lastportdynamic = 0;
      }
      for (DOM_Node inchild = child.getFirstChild();
	   inchild!=0;
	   inchild=inchild.getNextSibling()) {
	DOMString inchildname = inchild.getNodeName();
	if (inchildname.equals("file") && n->infiles) { 
	  file_node* newfile = new file_node;
	  newfile->description = NOT_SET;
	  newfile->datatype = NOT_SET;
	  ProcessFileNode(inchild,newfile);
	  n->infiles->insert(std::pair<int,file_node*>(n->infiles->size(),
							newfile));
	} else if (inchildname.equals("port") && n->inports) {
	  inport_node* newport = new inport_node;
	  newport->name = NOT_SET;
	  newport->description = NOT_SET;
	  newport->datatype = NOT_SET;
	  newport->upstream = new map<int,char*>;
	  ProcessInportNode(inchild,newport);
	  n->inports->insert(std::pair<int,inport_node*>(n->inports->size(),
							newport));
	} else if (inchildname.equals("device") && n->indevices) {
	  device_node* newdevice = new device_node;
	  newdevice->description = NOT_SET;
	  newdevice->devicename = NOT_SET;
	  ProcessDeviceNode(inchild,newdevice);
	  n->indevices->insert(std::pair<int,device_node*>(n->indevices->size(),newdevice));
	}
      }
    } else if (childname.equals("outputs")) {
      for (DOM_Node outchild = child.getFirstChild();
	   outchild!=0;
	   outchild=outchild.getNextSibling()) {
	DOMString outchildname = outchild.getNodeName();
	if (outchildname.equals("file") && n->outfiles) {
	  file_node* newfile = new file_node;
	  newfile->description = NOT_SET;
	  newfile->datatype = NOT_SET;
	  ProcessFileNode(outchild,newfile);
	  n->outfiles->insert(std::pair<int,file_node*>(n->outfiles->size(),
							newfile));
	} else if (outchildname.equals("port") && n->outports) {
	  outport_node* newport = new outport_node;
	  newport->name = NOT_SET;
	  newport->description = NOT_SET;
	  newport->datatype = NOT_SET;
	  newport->downstream = new map<int,char*>;
	  ProcessOutportNode(outchild,newport);
	  n->outports->insert(std::pair<int,outport_node*>(n->outports->size(),
							newport));
	} else if (outchildname.equals("device") && n->outdevices) {
	  device_node* newdevice = new device_node;
	  newdevice->description = NOT_SET;
	  newdevice->devicename = NOT_SET;
	  ProcessDeviceNode(outchild,newdevice);
	  n->outdevices->insert(std::pair<int,device_node*>(n->outdevices->size(),newdevice));
	}
      }
    }
  }
}

void ProcessParameterNode(const DOM_Node& d, parameter_node* n)
{
  for (DOM_Node child = d.getFirstChild();
       child!=0;
       child=child.getNextSibling()) {
    DOMString childname = child.getNodeName();
    if (childname.equals("label") && n->label==NOT_SET)
      n->label = rWSgSC(child);
    else if (childname.equals("widget") && n->widget==NOT_SET)
      n->widget = rWSgSC(child);
    else if (childname.equals("datatype") && n->datatype==NOT_SET)
      n->datatype = rWSgSC(child);
    else if (childname.equals("description") && n->description==NOT_SET)
      n->description = rWSgSC(child);
  }
}

void ProcessGuiNode(const DOM_Node& d, gui_node* n)
{
  for (DOM_Node child = d.getFirstChild();
       child!=0;
       child=child.getNextSibling()) {
    DOMString childname = child.getNodeName();
    if (childname.equals("description") && n->description==NOT_SET)
      n->description = rWSgSC(child);
    else if (childname.equals("img") && n->image==NOT_SET)
      n->image = rWSgSC(child);
    else if (childname.equals("parameter") && n->parameters) {
      parameter_node* newparam = new parameter_node;
      newparam->widget = NOT_SET;
      newparam->datatype = NOT_SET;
      newparam->label = NOT_SET;
      newparam->description = NOT_SET;
      ProcessParameterNode(child,newparam);
      n->parameters->insert(std::pair<int,parameter_node*>(n->parameters->size(),newparam));
    }
  }
}

void ProcessPlanNode(const DOM_Node& d, plan_node* n)
{
  for (DOM_Node child = d.getFirstChild();
       child!=0;
       child = child.getNextSibling()) {
    DOMString childname = child.getNodeName();
    if (childname.equals("description") && n->description==NOT_SET)
      n->description = rWSgSC(child);
    else if (childname.equals("step") && n->steps) 
      n->steps->insert(std::pair<int,char*>(n->steps->size(),
					    rWSgSC(child)));
  }
}

void ProcessOverviewNode(const DOM_Node& d, overview_node* n)
{
  for (DOM_Node child = d.getFirstChild();
       child!=0;
       child = child.getNextSibling()) {
    DOMString childname = child.getNodeName();
    if (childname.equals("authors") && n->authors) {
      for (DOM_Node author = child.getFirstChild();
           author!=0;
           author = author.getNextSibling()) {
        DOMString authorname = author.getNodeName();
        if (authorname.equals("author"))
          n->authors->insert(std::pair<int,char*>(n->authors->size(),
            author.getFirstChild().getNodeValue().transcode()));
      }
    }
    else if (childname.equals("description") && n->description==NOT_SET) {
      n->description = rWSgSC(child);
    } else if (childname.equals("summary") && n->summary==NOT_SET)
      n->summary = rWSgSC(child);
    else if (childname.equals("examplesr") && n->examplesr==NOT_SET)
      n->examplesr = rWSgSC(child);
  }
}

void ProcessImplementationNode(const DOM_Node& d, implementation_node* n)
{
  for (DOM_Node child = d.getFirstChild();
       child!=0;
       child=child.getNextSibling()) {
    DOMString childname = child.getNodeName();
    if (childname.equals("ccfile") && n->ccfiles) { 
      n->ccfiles->insert(std::pair<int,char*>(n->ccfiles->size(),rWSgSC(child)));
    } else if (childname.equals("cfile") && n->cfiles) {
      n->cfiles->insert(std::pair<int,char*>(n->cfiles->size(),rWSgSC(child)));
    } else if (childname.equals("ffile") && n->ffiles) {
      n->ffiles->insert(std::pair<int,char*>(n->ffiles->size(),rWSgSC(child)));
    }
  }
}

void ProcessComponentNode(const DOM_Node& d, component_node* n)
{
  if (n->name==NOT_SET) {
    DOM_Node name = d.getAttributes().getNamedItem("name");
    if (name==0) 
      cout << "ERROR: Component has no name." << endl;
    else {
      n->name = name.getNodeValue().transcode();
    }
  }
  
  if (n->category==NOT_SET) {
    DOM_Node name = d.getAttributes().getNamedItem("category");
    if (name==0)
      cout << "ERROR: Component has no category." << endl;
    else {
      n->category = name.getNodeValue().transcode();
    }
  }

  //cout << "processing " << n->category << "::" << n->name << endl;
  
  for (DOM_Node child = d.getFirstChild();
       child!=0;
       child = child.getNextSibling()) {
    DOMString childname = child.getNodeName();
    if (childname.equals("overview") && n->overview)
      ProcessOverviewNode(child,n->overview);

    if (childname.equals("implementation") && n->implementation)
      ProcessImplementationNode(child,n->implementation);
    
    if (childname.equals("io") && n->io) 
      ProcessIoNode(child,n->io);
    
    if (childname.equals("gui") && n->gui)
      ProcessGuiNode(child,n->gui);
    
    if (childname.equals("testing") && n->testingplans) {
      for (DOM_Node planchild = child.getFirstChild();
	   planchild!=0;
	   planchild = planchild.getNextSibling()) {
	DOMString planchildname = planchild.getNodeName();
	if (planchildname.equals("plan")) {
	  plan_node* newplan = new plan_node;
	  newplan->description = NOT_SET;
	  newplan->steps = new map<int,char*>;
	  ProcessPlanNode(planchild,newplan);
	  n->testingplans->insert(std::pair<int,plan_node*>(n->testingplans->size(),newplan));
	}
      }
    }
  }
}

void WriteOverviewNodeToStream(overview_node* n, std::ofstream& o)
{
  o << "  <overview>" << endl;
  
  if (n->authors) {
    o << "    <authors>" << endl;
    for (char_iter i=n->authors->begin();
	 i!=n->authors->end();
	 i++)
      o << "      <author>" << (*i).second << "</author>" << endl;
    o << "    </authors>" << endl;
  }

  if (n->summary && n->summary!=NOT_SET)
    o << "    <summary>" << endl
      << "      " << n->summary << endl
      << "    </summary>" << endl;
    
  if (n->description && n->description!=NOT_SET)
    o << "    <description>" << endl
      << "      " << n->description << endl
      << "    </description>" << endl;

  if (n->examplesr && n->examplesr!=NOT_SET)
    o << "    <examplesr>" << n->examplesr << "</examplesr>" << endl;

  o << "  </overview>" << endl;
}

void WriteFileNodeToStream(file_node* n, std::ofstream& o)
{
  o << "      <file>" << endl;

  if (n->description && n->description!=NOT_SET)
    o << "        <description>" << endl 
      << "          " << n->description << endl
      << "        </description>" << endl;

  if (n->datatype && n->datatype!=NOT_SET)
    o << "        <datatype>" << n->datatype
      << "</datatype>" << endl;

  o << "      </file>" << endl;
}

void WriteDeviceNodeToStream(device_node* n, std::ofstream& o)
{
  o << "      <device>" << endl;
    
  if (n->devicename && n->devicename!=NOT_SET)
    o << "        <devicename>" << n->devicename
      << "</devicename>" << endl;
  
  if (n->description && n->description!=NOT_SET)
    o << "        <description>" << endl 
      << "          " << n->description << endl
      << "        </description>" << endl;

  o << "      </device>" << endl;
}

void WriteInportNodeToStream(inport_node* n, std::ofstream& o)
{
  char_iter i;

  o << "      <port>" << endl;

  if (n->name && n->name!=NOT_SET)
    o << "        <name>" << n->name
      << "</name>" << endl;

  if (n->description && n->description!=NOT_SET)
    o << "        <description>" << endl 
      << "          " << n->description << endl
      << "        </description>" << endl;

  if (n->datatype && n->datatype!=NOT_SET)
    o << "        <datatype>" << n->datatype
      << "</datatype>" << endl;

  if (n->upstream)
    for (i=n->upstream->begin();
	 i!=n->upstream->end();
	 i++)
      o <<  "        <componentname>" << (*i).second
	<< "</componentname>" << endl;
  
  o << "      </port>" << endl;
}

void WriteOutportNodeToStream(outport_node* n, std::ofstream& o)
{
  char_iter i;

  o << "      <port>" << endl;

  if (n->name && n->name!=NOT_SET)
    o << "        <name>" << n->name
      << "</name>" << endl;

  if (n->description && n->description!=NOT_SET)
    o << "        <description>" << endl 
      << "          " << n->description << endl
      << "        </description>" << endl;

  if (n->datatype && n->datatype!=NOT_SET)
    o << "        <datatype>" << n->datatype
      << "</datatype>" << endl;

  if (n->downstream)
    for (i=n->downstream->begin();
	 i!=n->downstream->end();
	 i++)
      o <<  "        <componentname>" << (*i).second 
	<< "</componentname>" << endl;

  o << "      </port>" << endl;
}

void WriteIoNodeToStream(io_node* n, std::ofstream& o)
{
  file_iter i2;
  inport_iter i5;
  outport_iter i6;
  device_iter i7;

  if (!n->infiles && !n->inports && !n->indevices &&
      !n->outfiles && !n->outports && !n->outdevices)
    return;

  o << "  <io>" << endl;

  if (n->infiles || n->inports || n->indevices) {
    o << "    <inputs lastportdynamic=\"";
    if (n->lastportdynamic)
      o << "yes\">" << endl;
    else
      o << "no\">" << endl;

    if (n->infiles)
      for (i2=n->infiles->begin();
	   i2!=n->infiles->end();
	   i2++)
	WriteFileNodeToStream((*i2).second,o);
    
    if (n->indevices)
      for (i7=n->indevices->begin();
	   i7!=n->indevices->end();
	   i7++)
	WriteDeviceNodeToStream((*i7).second,o);

    if (n->inports)
      for (i5=n->inports->begin();
	   i5!=n->inports->end();
	   i5++)
	WriteInportNodeToStream((*i5).second,o);

    o << "    </inputs>" << endl;
  }

  if (n->outfiles || n->outports || n->outdevices) {
    o << "    <outputs>" << endl;

    if (n->outfiles)
      for (i2=n->outfiles->begin();
	   i2!=n->outfiles->end();
	   i2++)
	WriteFileNodeToStream((*i2).second,o);
    
    if (n->outdevices)
      for (i7=n->outdevices->begin();
	   i7!=n->outdevices->end();
	   i7++)
	WriteDeviceNodeToStream((*i7).second,o);

    if (n->outports)
      for (i6=n->outports->begin();
	   i6!=n->outports->end();
	   i6++) 
	WriteOutportNodeToStream((*i6).second,o);

    o << "    </outputs>" << endl;
  }

  o << "  </io>" << endl;
}

void WriteParameterNodeToStream(parameter_node* n, std::ofstream& o)
{
  o << "    <parameter>" << endl;

  if (n->widget && n->widget!=NOT_SET)
    o << "      <widget>" << n->widget << "</widget>" << endl;

  if (n->label && n->label!=NOT_SET)
    o << "      <label>" << n->label << "</label>" << endl;
  
  if (n->datatype && n->datatype!=NOT_SET)
    o << "      <datatype>" << n->datatype << "</datatype>" << endl;
  
  if (n->description && n->description!=NOT_SET)
    o << "      <description>" << endl
      << "        " << n->description << endl
      << "      </description>" << endl;

  o << "    </parameter>" << endl;
}

void WriteGuiNodeToStream(gui_node* n, std::ofstream& o)
{
  param_iter i2;

  o << "  <gui>" << endl;

  if (n->description && n->description!=NOT_SET)
    o << "    <description>" << endl
      << "      " << n->description << endl
      << "    </description>" << endl;
  
  if (n->parameters) {
    for (i2=n->parameters->begin();
	 i2!=n->parameters->end();
	 i2++)
      WriteParameterNodeToStream((*i2).second,o);
  }
  
  if (n->image && n->image!=NOT_SET)
    o << "    <img>" << n->image << "</img>" << endl;
  
  o << "  </gui>" << endl;
}

void WritePlanNodeToStream(plan_node* n, std::ofstream& o)
{
  char_iter i;

  o << "    <plan>" << endl;

  if (n->description && n->description!=NOT_SET)
    o << "      <description>" << endl
      << "        " << n->description << endl
      << "      </description>" << endl;

  if (n->steps) 
    for (i=n->steps->begin();
	 i!=n->steps->end();
	 i++)
      o << "      <step>" << endl
	<< "        " << (*i).second << endl
	<< "      </step>" << endl;

  o << "    </plan>" << endl;
}

void WriteImplementationNodeToStream(implementation_node* n, std::ofstream& o)
{
  char_iter i;

  if (n->ccfiles || n->cfiles || n->ffiles) {
    o << "  <implementation>" << endl;
    
    for (i=n->ccfiles->begin();
	 i!=n->ccfiles->end();
	 i++)
      o << "    <ccfile>" << (*i).second << "</ccfile>" << endl;

    for (i=n->cfiles->begin();
	 i!=n->cfiles->end();
	 i++)
      o << "    <cfile>" << (*i).second << "</cfile>" << endl;

    for (i=n->ffiles->begin();
	 i!=n->ffiles->end();
	 i++)
      o << "    <ffile>" << (*i).second << "</ffile>" << endl;

    o << "  </implementation>" << endl;
  }
}

void WriteComponentNodeToFile(component_node* n, const char* filename)
{
  std::ofstream o(filename);

  o << "<?xml version=\"1.0\" encoding=\"UTF-8\" ?>" << endl;
  o << "<!DOCTYPE component SYSTEM \""
       "../../../../../doc/component.dtd\">" << endl;
  o << "<?xml-stylesheet href=\"../../../../../doc/component.xsl\""
       " type=\"text/xsl\"?>" << endl;
  o << "<?cocoon-process type=\"xslt\"?>" << endl << endl;

  o << "<component name=\"";
  if (n->name && n->name!=NOT_SET)
    o << n->name << "\" category=\"";
  else
    o << "\" category=\"";

  if (n->category && n->category!=NOT_SET)
    o << n->category << "\">" << endl;

  if (n->overview)
    WriteOverviewNodeToStream(n->overview,o);

  if (n->implementation)
    WriteImplementationNodeToStream(n->implementation,o);

  if (n->io)
    WriteIoNodeToStream(n->io,o);

  if (n->gui)
    WriteGuiNodeToStream(n->gui,o);
  
  if (n->testingplans) {
    o << "  <testing>" << endl;
    for(plan_iter i=n->testingplans->begin();
	i!=n->testingplans->end();
	i++)
      WritePlanNodeToStream((*i).second,o);
    o << "  </testing>" << endl;
  }

    o << "</component>" << endl;
  
  o << endl;
}

int ReadComponentNodeFromFile(component_node* n, const char* filename)
{
  // Initialize the XML4C system
  try {
    XMLPlatformUtils::Initialize();
  } catch (const XMLException& toCatch) {
    std::cerr << "Error during initialization! :\n"
	 << StrX(toCatch.getMessage()) << endl;
    return -1;
  }

  // Instantiate the DOM parser.
  DOMParser parser;
  parser.setDoValidation(false);
  
  PackageDBHandler handler;
  parser.setErrorHandler(&handler);
  
  try {
    parser.parse(filename);
  }  catch (const XMLException& toCatch) {
    std::cerr << "Error during parsing: '" <<
      filename << "'\nException message is:  " <<
      xmlto_string(toCatch.getMessage());
    handler.foundError=true;
    return 0;
  }
  
  DOM_Document doc = parser.getDocument();
  DOM_NodeList list = doc.getElementsByTagName("component");
  int nlist = list.getLength();
  if (nlist==0) return 0;
  for (int i=0;i<nlist;i++)
    ProcessComponentNode(list.item(i),n);
  return 1;
}

} // End namespace SCIRun


