/* REFERENCED */
static char *id="@(#) $Id$";

#include "DataWarehouseException.h"

namespace Uintah {
namespace Exceptions {

DataWarehouseException::DataWarehouseException(const std::string& msg)
    : msg(msg)
{
}

std::string DataWarehouseException::message() const
{
    return msg;
}

} // end namespace Exception
} // end namespace Uintah

//
// $Log$
// Revision 1.2  2000/03/17 18:45:40  dav
// fixed a few more namespace problems
//
//
