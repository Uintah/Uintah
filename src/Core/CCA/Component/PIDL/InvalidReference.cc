
/*
 *  InvalidReference.h: A "bad" reference to an object
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <Core/CCA/Component/PIDL/InvalidReference.h>

using PIDL::InvalidReference;

InvalidReference::InvalidReference(const std::string& msg)
    : d_msg(msg)
{
}

InvalidReference::InvalidReference(const InvalidReference& copy)
    : d_msg(copy.d_msg)
{
}

InvalidReference::~InvalidReference()
{
}

const char* InvalidReference::message() const
{
    return d_msg.c_str();
}

const char* InvalidReference::type() const
{
    return "Core/CCA/Component::PIDL::InvalidReference";
}

