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
 *  GuiVar.cc: Interface to TCL variables
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   September 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Core/GuiInterface/GuiVar.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
using namespace SCIRun;

#include <iostream>
using namespace std;

GuiVar::GuiVar(GuiContext* ctx)
  : ctx(ctx)
{
}

GuiVar::~GuiVar()
{
}

void GuiVar::reset()
{
  ctx->reset();
}

template class GuiSingle<string>;
template class GuiSingle<double>;
template class GuiSingle<int>;
template class GuiTriple<Point>;
template class GuiTriple<Vector>;


