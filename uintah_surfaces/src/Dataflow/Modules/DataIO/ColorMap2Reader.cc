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
 *  ColorMap2Reader.cc: Read a persistent colormap from a file
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   Sept 2004
 *
 *  Copyright (C) 2004 SCI Group
 */

#include <Dataflow/Ports/Colormap2Port.h>
#include <Dataflow/Modules/DataIO/GenericReader.h>
//#include <Core/ImportExport/ColorMap/ColorMapIEPlugin.h>

namespace SCIRun {

template class GenericReader<ColorMap2Handle>;

class ColorMap2Reader : public GenericReader<ColorMap2Handle> {
protected:
  GuiString gui_types_;
  GuiString gui_filetype_;

  virtual bool call_importer(const string &filename);

public:
  ColorMap2Reader(GuiContext* ctx);

  virtual void execute();
};

DECLARE_MAKER(ColorMap2Reader)

ColorMap2Reader::ColorMap2Reader(GuiContext* ctx)
  : GenericReader<ColorMap2Handle>("ColorMap2Reader", ctx, "DataIO", "SCIRun"),
    gui_types_(ctx->subVar("types", false)),
    gui_filetype_(ctx->subVar("filetype"))
{
  string importtypes = "{";
  importtypes += "{{SCIRun ColorMap2 File} {.cmap2} } ";
  importtypes += "{{SCIRun ColorMap2 Any} {.*} } ";
  importtypes += "}";

  gui_types_.set(importtypes);
}


bool
ColorMap2Reader::call_importer(const string &/*filename*/)
{
  return false;
}


void
ColorMap2Reader::execute()
{
  importing_ = false;
  GenericReader<ColorMap2Handle>::execute();
}


} // End namespace SCIRun
