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


#include <Dataflow/Network/PackageDBHandler.h>

#include <Core/Containers/StringUtil.h>
#include <Core/GuiInterface/GuiInterface.h>
#include <Core/XMLUtil/XMLUtil.h>
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
