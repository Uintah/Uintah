/* REFERENCED */
static char *id="@(#) $Id$";

#include "DataWarehouseException.h"

namespace Uintah {
namespace Exceptions {

DataWarehouseException::DataWarehouseException(const std::string& msg)
    : msg(msg)
{
}

const char* DataWarehouseException::message() const
{
    return msg.c_str();
}

} // end namespace Exception
} // end namespace Uintah

//
// $Log$
// Revision 1.3  2000/03/22 23:41:24  sparker
// Working towards getting arches to compile/run
//
// Revision 1.2  2000/03/17 18:45:40  dav
// fixed a few more namespace problems
//
//
