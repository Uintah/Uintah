
/*
 *  GlobusError.h: Erros due to globus calls
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <Core/CCA/Component/PIDL/GlobusError.h>
#include <sstream>

using Component::PIDL::GlobusError;

GlobusError::GlobusError(const std::string& msg, int code)
    : d_msg(msg), d_code(code)
{
    std::ostringstream o;
    o << d_msg << "(return code=" << d_code << ")";
    d_msg = o.str();
}

GlobusError::GlobusError(const GlobusError& copy)
    : d_msg(copy.d_msg), d_code(copy.d_code)
{
}

GlobusError::~GlobusError()
{
}

const char* GlobusError::message() const
{
    return d_msg.c_str();
}

const char* GlobusError::type() const
{
    return "Core/CCA/Component::PIDL::GlobusError";
}

