/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
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







