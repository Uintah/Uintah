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



#include <Core/CCA/Exceptions/CommError.h>
#include <sci_defs/framework_defs.h>

#include <iostream>
#include <sstream>

using namespace SCIRun;

CommError::CommError(const std::string& msg, const char* file, int line, int errorCode)
    : d_msg(msg), d_code(errorCode)
{
    std::ostringstream o;
    o << d_msg << " (error code=" << d_code << ")";
    d_msg = o.str();
#ifdef FWK_DEBUG
    std::cerr << "CommError: " << d_msg << " thrown in\n" << file << ":" << line << std::endl;
#endif
}

CommError::CommError(const CommError& copy)
    : d_msg(copy.d_msg), d_code(copy.d_code)
{
}

CommError::~CommError()
{
}

const char* CommError::message() const
{
    return d_msg.c_str();
}

const char* CommError::type() const
{
    return "PIDL::CommError";
}
