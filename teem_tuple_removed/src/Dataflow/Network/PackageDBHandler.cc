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

#include <Dataflow/Network/PackageDBHandler.h>

#include <Core/Containers/StringUtil.h>
#include <Core/GuiInterface/GuiInterface.h>
#include <Dataflow/XMLUtil/XMLUtil.h>
#include <Dataflow/Network/NetworkEditor.h>

namespace SCIRun {


PackageDBHandler::PackageDBHandler(GuiInterface* gui)
  : gui(gui)
{
  foundError=false;
}

PackageDBHandler::~PackageDBHandler()
{
}

void PackageDBHandler::error(const SAXParseException& e)
{
  foundError=true;
  gui->postMessage(string("Error at (file ")+xmlto_string(e.getSystemId())
		   +", line "+to_string((int)e.getLineNumber())
		   +", char "+to_string((int)e.getColumnNumber())
		   +"): "+xmlto_string(e.getMessage()));
}

void PackageDBHandler::fatalError(const SAXParseException& e)
{
  foundError=true;
  gui->postMessage(string("Fatal Error at (file ")+xmlto_string(e.getSystemId())
		   +", line "+to_string((int)e.getLineNumber())
		   +", char "+to_string((int)e.getColumnNumber())
		   +"): "+xmlto_string(e.getMessage()));
}

void PackageDBHandler::warning(const SAXParseException& e)
{
  gui->postMessage(string("Warning at (file ")+xmlto_string(e.getSystemId())
		   +", line "+to_string((int)e.getLineNumber())
		   +", char "+to_string((int)e.getColumnNumber())
		   +"): "+xmlto_string(e.getMessage()));
}

void PackageDBHandler::resetErrors()
{
}

} // End namespace SCIRun
