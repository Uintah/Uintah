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
 *  ColorMapReader.cc: Read a persistent colormap from a file
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Dataflow/Ports/ColorMapPort.h>
#include <Dataflow/Modules/DataIO/GenericReader.h>

namespace SCIRun {

template class GenericReader<ColorMapHandle>;

class ColorMapReader : public GenericReader<ColorMapHandle> {
public:
  ColorMapReader(GuiContext* ctx);
};

DECLARE_MAKER(ColorMapReader)

ColorMapReader::ColorMapReader(GuiContext* ctx)
  : GenericReader<ColorMapHandle>("ColorMapReader", ctx, "DataIO", "SCIRun")
{
}

} // End namespace SCIRun
