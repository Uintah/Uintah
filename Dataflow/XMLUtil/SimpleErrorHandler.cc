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


#include <Dataflow/XMLUtil/SimpleErrorHandler.h>
#include <Dataflow/XMLUtil/XMLUtil.h>
#include <Core/Containers/StringUtil.h>
#include <iostream>
#ifndef __sgi
#include <stdio.h>
#endif
using namespace std;

namespace SCIRun {

SimpleErrorHandler::SimpleErrorHandler()
{
  foundError=false;
}

SimpleErrorHandler::~SimpleErrorHandler()
{
}

static string sehToString(int i)
{
  char buf[20];
  sprintf(buf, "%d", i);
  return string(buf);
}

static void postMessage(const string& errmsg)
{
  cerr << errmsg << '\n';
}

void SimpleErrorHandler::error(const SAXParseException& e)
{
  foundError=true;
  postMessage("Error at (file " + xmlto_string(e.getSystemId())
	      + ", line " + to_string((int)e.getLineNumber())
	      + ", char " + to_string((int)e.getColumnNumber())
	      + "): " + xmlto_string(e.getMessage()));
}

void SimpleErrorHandler::fatalError(const SAXParseException& e)
{
  foundError=true;
  postMessage("Fatal Error at (file " + xmlto_string(e.getSystemId())
	      + ", line " + sehToString((int)e.getLineNumber())
	      + ", char " + sehToString((int)e.getColumnNumber())
	      + "): " + xmlto_string(e.getMessage()));
}

void SimpleErrorHandler::warning(const SAXParseException& e)
{
  postMessage("Warning at (file " + xmlto_string(e.getSystemId())
	      + ", line " + sehToString((int)e.getLineNumber())
	      + ", char " + sehToString((int)e.getColumnNumber())
	      + "): " + xmlto_string(e.getMessage()));
}

void SimpleErrorHandler::resetErrors()
{
}

} // End namespace SCIRun







