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

/*
 *  SCIRunErrorHandler.cc:
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#include <SCIRun/SCIRunErrorHandler.h>
#include <Core/Containers/StringUtil.h>
#include <Dataflow/XMLUtil/XMLUtil.h>
#include <iostream>
using namespace std;
using namespace SCIRun;

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#define IRIX
#pragma set woff 1375
#endif
#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/sax/SAXException.hpp>
#include <xercesc/sax/SAXParseException.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/dom/DOMNamedNodeMap.hpp>
#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma reset woff 1375
#endif

SCIRunErrorHandler::SCIRunErrorHandler()
{
  foundError=false;
}

SCIRunErrorHandler::~SCIRunErrorHandler()
{
}

void SCIRunErrorHandler::postMessage(const std::string& message)
{
  cerr << message << '\n';
}

void SCIRunErrorHandler::error(const SAXParseException& e)
{
  foundError=true;
  postMessage(string("Error at (file ")+xmlto_string(e.getSystemId())
	      +", line "+to_string((int)e.getLineNumber())
	      +", char "+to_string((int)e.getColumnNumber())
	      +"): "+xmlto_string(e.getMessage()));
}

void SCIRunErrorHandler::fatalError(const SAXParseException& e)
{
  foundError=true;
  postMessage(string("Fatal Error at (file ")+xmlto_string(e.getSystemId())
	      +", line "+to_string((int)e.getLineNumber())
	      +", char "+to_string((int)e.getColumnNumber())
	      +"): "+xmlto_string(e.getMessage()));
}

void SCIRunErrorHandler::warning(const SAXParseException& e)
{
  postMessage(string("Warning at (file ")+xmlto_string(e.getSystemId())
	      +", line "+to_string((int)e.getLineNumber())
	      +", char "+to_string((int)e.getColumnNumber())
	      +"): "+xmlto_string(e.getMessage()));
}

void SCIRunErrorHandler::resetErrors()
{
}

