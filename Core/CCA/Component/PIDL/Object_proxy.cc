
/*
 *  Object.h: Base class for all PIDL distributed objects
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <Core/CCA/Component/PIDL/Object_proxy.h>
#include <Core/CCA/Component/PIDL/GlobusError.h>
#include <Core/CCA/Component/PIDL/TypeInfo.h>
#include <Core/CCA/Component/PIDL/URL.h>
#include <iostream>
#include <string>

using Component::PIDL::GlobusError;
using Component::PIDL::Object_proxy;
using Component::PIDL::TypeInfo;

Object_proxy::Object_proxy(const Reference& ref)
    : ProxyBase(ref)
{
}

Object_proxy::Object_proxy(const URL& url)
    : ProxyBase(Reference())
{
    std::string s(url.getString());
    d_ref.d_vtable_base=TypeInfo::vtable_methods_start;
    char* str=const_cast<char*>(s.c_str());
    if(int gerr=globus_nexus_attach(str, &d_ref.d_sp)){
	d_ref.d_vtable_base=TypeInfo::vtable_invalid;
	throw GlobusError("nexus_attach", gerr);
    }
}

Object_proxy::~Object_proxy()
{
}

