/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */
/*
 *  String.cc:  String Object
 *
 *  Written by:
 *   Jeroen Stinstra
 *   Department of Computer Science
 *   University of Utah
 *   October 2005
 *
 */


#include <Core/Datatypes/String.h>
#include <Core/Malloc/Allocator.h>

#include <string>

namespace SCIRun {

static Persistent*
maker()
{
  return scinew String;
}

PersistentTypeID String::type_id("String", "PropertyManager", maker);


//! constructors
String::String()
{
}

String::String(const String& s) :
  str_(s.str_)
{
}

String::String(const std::string& s) :
  str_(s)
{
}

String::String(const char* s) :
  str_(s)
{
}

String::~String()
{
}

String*
String::clone()
{
  return scinew String(*this);
}

#define STRING_VERSION 1

void
String::io(Piostream& stream)
{
  /*int version=*/stream.begin_class("String", STRING_VERSION);

  // Do the base class first.
  PropertyManager::io(stream);

  stream.begin_cheap_delim();
  stream.io(str_);
  stream.end_cheap_delim();
  stream.end_class();
}

} // End namespace SCIRun
