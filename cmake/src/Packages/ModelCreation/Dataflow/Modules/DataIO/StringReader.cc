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

/*
 *  StringReader.cc:
 *
 *  Written by:
 *   jeroen
 *   TODAY'S DATE HERE
 *
 */

#include <sgi_stl_warnings_off.h>
#include <fstream>
#include <string>
#include <sgi_stl_warnings_on.h>

#include <Dataflow/Network/Ports/StringPort.h>
#include <Packages/ModelCreation/Dataflow/Modules/DataIO/GenericReader.h>

namespace ModelCreation {

using namespace SCIRun;

class StringReader : public GenericReader<StringHandle> {
public:
  StringReader(GuiContext*);
  virtual ~StringReader();

  virtual void execute();
  
protected:
  GuiString gui_types_;

  virtual bool call_importer(const string &filename);
    
};

DECLARE_MAKER(StringReader)

StringReader::StringReader(GuiContext* ctx)
  : GenericReader<StringHandle>("StringReader", ctx, "DataIO", "ModelCreation"),
    gui_types_(get_ctx()->subVar("types", false))
{
  gui_types_.set("{ {{textfile} {.txt .asc .doc}} {{all files} {.*}} }");
}


StringReader::~StringReader()
{
}


void
StringReader::execute()
{
  importing_ = true;
  GenericReader<StringHandle>::execute();
}


// Simple text reader
bool
StringReader::call_importer(const string &filename)
{
  std::string input;
  try
  {
    std::ifstream file(filename.c_str());
    std::string line;

    getline(file,line);
    input += line;
           
    while(!file.eof())
    { 
      getline(file,line);
      input += "\n";
      input += line; 
    }   
  }
  catch(...)
  {
    error("Could not open file: " + filename);
    return(false);
  }
  
  handle_ = scinew String(input);
  return true;
}


} // End namespace ModelCreation


