
/*
 *  PickMessage.h: Messages back to Modules about pick info
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Geom/PickMessage.h>

GeomPickMessage::GeomPickMessage(Module* module, void* cbdata)
: MessageBase(MessageTypes::GeometryPick),
  module(module), cbdata(cbdata)
{
}

GeomPickMessage::GeomPickMessage(Module* module, void* cbdata, int)
: MessageBase(MessageTypes::GeometryRelease),
  module(module), cbdata(cbdata)
{
}

GeomPickMessage::GeomPickMessage(Module* module, int axis, double distance,
				 const Vector& delta, void* cbdata)
: MessageBase(MessageTypes::GeometryPick),
  module(module), axis(axis), distance(distance), delta(delta), cbdata(cbdata)
{
}

GeomPickMessage::~GeomPickMessage()
{
}
