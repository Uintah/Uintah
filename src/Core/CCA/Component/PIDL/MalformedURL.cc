
/*
 *  MalformedURL.h: Base class for PIDL Exceptions
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <Core/CCA/Component/PIDL/MalformedURL.h>

using PIDL::MalformedURL;

MalformedURL::MalformedURL(const std::string& url,
			   const std::string& error)
    : d_url(url), d_error(error)
{
    d_msg = "Malformed URL: "+d_url+" ("+d_error+")"; 
}

MalformedURL::MalformedURL(const MalformedURL& copy)
    : d_url(copy.d_url), d_error(copy.d_error)
{
}

MalformedURL::~MalformedURL()
{
}

const char* MalformedURL::message() const
{
    return d_msg.c_str();
}

const char* MalformedURL::type() const
{
    return "Core/CCA/Component::PIDL::MalformedURL";
}

