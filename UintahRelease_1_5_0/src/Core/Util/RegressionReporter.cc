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


#include <Core/Util/RegressionReporter.h>

namespace SCIRun {

RegressionReporter::RegressionReporter() :
  log_("regression.log")
{
}

RegressionReporter::RegressionReporter(std::string logname) :
  log_(logname)
{
}

void
RegressionReporter::error(const std::string& msg)
{
  std::cout << "REGRESSION TEST ERROR: " << msg << std::endl;
  log_.putmsg(std::string("REGRESSION TEST ERROR: ") + msg);
}


void
RegressionReporter::warning(const std::string& msg)
{
  log_.putmsg(std::string("REGRESSION TEST WARNING: ") + msg);
}


void
RegressionReporter::remark(const std::string& msg)
{
  log_.putmsg(std::string("REGRESSION TEST REMARK: ") + msg);
}


void 
RegressionReporter::compile_error(const std::string &filename)
{
  std::cerr << "REGRESSION TEST DYNAMIC COMPILE FAILURE IN FILE: " << filename << "cc" << std::endl;
  log_.putmsg(std::string("REGRESSION TEST DYNAMIC COMPILE FAILURE IN FILE: ") + filename + "cc");
}


void
RegressionReporter::add_raw_message(const std::string& msg)
{
  log_.putmsg(msg);
}


void
RegressionReporter::regression_message(const std::string& msg)
{
  std::cout << msg << std::endl;
}


void
RegressionReporter::regression_error(const std::string& msg)
{
  std::cout << "ERROR: " << msg << std::endl;
}


std::ostream &
RegressionReporter::msg_stream()
{
  return std::cout;
}


void
RegressionReporter::msg_stream_flush()
{
}


void
RegressionReporter::report_progress( ProgressState )
{
}


void
RegressionReporter::update_progress(double)
{
}

void
RegressionReporter::update_progress(int current, int max)
{
}

void
RegressionReporter::increment_progress()
{
}


} // End namespace SCIRun
