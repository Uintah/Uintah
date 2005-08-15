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
 *  Connection.cc: A Connection between two modules
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <iostream>
#include <stdio.h>
#ifndef _WIN32
#include <unistd.h>
#else
#include <io.h>
#endif

using std::cerr;
using std::endl;

#include <Dataflow/Network/Connection.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Network/Port.h>
#include <Core/Math/MinMax.h>
#include <Core/Math/MiscMath.h>
#include <Core/GuiInterface/GuiContext.h>
#include <Core/Containers/StringUtil.h>

using namespace SCIRun;

Connection::Connection(Module* m1, int p1, Module* m2, int p2,
		       const string &id)
  : oport(m1->getOPort(p1)),
    iport(m2->getIPort(p2)),
    id(id),
    disabled_(false)
{
  iport->attach(this);
  oport->attach(this);
  makeID();
}

Connection::~Connection()
{
  oport->detach(this, false);
  iport->detach(this, disabled_);
}

void Connection::makeID()
{
  if (!oport || !iport || !oport->get_module() || !iport->get_module()) return;
  oport->get_module()->getGui()->eval("makeConnID {"+
				      oport->get_module()->getID()+" "+
				      to_string(oport->get_which_port())+" "+
				      iport->get_module()->getID()+" "+
				      to_string(iport->get_which_port())+"}",
				      id);
}
