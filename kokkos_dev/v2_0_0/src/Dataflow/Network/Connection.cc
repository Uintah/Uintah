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

Connection::Connection(Module* m1, int p1, Module* m2, int p2) :
  blocked_(false)
{
  oport=m1->getOPort(p1);
  iport=m2->getIPort(p2);
}

void Connection::connect()
{
  iport->attach(this);
  oport->attach(this);
  makeID();
}

Connection::~Connection()
{
  oport->detach(this);
  iport->detach(this);
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
