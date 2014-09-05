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


// Service.cc

#include <Core/Services/Service.h>

namespace SCIRun {


Service::Service(ServiceContext &ctx) :
	lock("Service Lock"),
	ref_cnt(0),
	ctx_(ctx)
{
}

Service::~Service()
{
	// Close the socket if it was not closed yet
	// This function should be thread-safe

	ctx_.socket.close();

}


void Service::run()
{
	try
    {
        execute();	
    }
    catch (...)
    {
        ctx_.socket.close();
        throw;
    }
	ctx_.socket.close();
}

void Service::execute()
{
}

void Service::errormsg(std::string error)
{
	ctx_.log->putmsg(std::string("Service (" + ctx_.servicename + std::string(") : error = ") + error));
}

void Service::warningmsg(std::string warning)
{
	ctx_.log->putmsg(std::string("Service (" + ctx_.servicename + std::string(") : warning = ") + warning));
}

} // end namespace
