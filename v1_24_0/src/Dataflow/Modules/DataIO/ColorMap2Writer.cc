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
 *  ColorMap2Writer.cc: Save persistent representation of a colormap to a file
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   September 2004
 *
 *  Copyright (C) 2004 SCI Group
 */

#include <Dataflow/Ports/Colormap2Port.h>
#include <Dataflow/Modules/DataIO/GenericWriter.h>
//#include <Core/ImportExport/ColorMap/ColorMapIEPlugin.h>

namespace SCIRun {

template class GenericWriter<ColorMap2Handle>;

class ColorMap2Writer : public GenericWriter<ColorMap2Handle> {
protected:
  GuiString gui_types_;
  GuiString gui_exporttype_;

  virtual bool call_exporter(const string &filename);

public:
  ColorMap2Writer(GuiContext* ctx);

  virtual void execute();
};


DECLARE_MAKER(ColorMap2Writer)

ColorMap2Writer::ColorMap2Writer(GuiContext* ctx)
  : GenericWriter<ColorMap2Handle>("ColorMap2Writer", ctx, "DataIO", "SCIRun"),
    gui_types_(ctx->subVar("types", false)),
    gui_exporttype_(ctx->subVar("exporttype"))
{
  string exporttypes = "{";
  exporttypes += "{{SCIRun ColorMap2 Binary} {.cmap} } ";
  exporttypes += "{{SCIRun ColorMap2 ASCII} {.cmap} } ";
  exporttypes += "}";

  gui_types_.set(exporttypes);
}


bool
ColorMap2Writer::call_exporter(const string &/*filename*/)
{
  return false;
}



void
ColorMap2Writer::execute()
{
  const string ftpre = gui_exporttype_.get();
  const string::size_type loc = ftpre.find(" (");
  const string ft = ftpre.substr(0, loc);

  exporting_ = false;

  // Determine if we're ASCII or Binary
  string ab = "Binary";
  if (ft == "SCIRun ColorMap2 ASCII") ab = "ASCII";
  filetype_.set(ab);

  GenericWriter<ColorMap2Handle>::execute();
}



} // End namespace SCIRun
