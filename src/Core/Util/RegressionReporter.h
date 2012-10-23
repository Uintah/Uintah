/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
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



#ifndef SCIRun_Core_Util_RegressionReporter_h
#define SCIRun_Core_Util_RegressionReporter_h

#include <Core/Util/LogFile.h>
#include <Core/Util/ProgressReporter.h>
#include <Core/Util/share.h>

namespace SCIRun {

class SCISHARE RegressionReporter : public ProgressReporter 
{
public:
  RegressionReporter();
  RegressionReporter(std::string logfile);

  virtual void          error(const std::string& msg);
  virtual void          warning(const std::string& msg);
  virtual void          remark(const std::string& msg);
  virtual void          compile_error(const std::string &filename);
  virtual void          add_raw_message(const std::string &msg);

  virtual void          regression_message(const std::string& msg);
  virtual void          regression_error(const std::string& msg);

  // This one isn't as thread safe as the other ProgressReporter functions.
  // Use add_raw_message or one of the others instead if possible.
  virtual std::ostream &msg_stream();
  virtual void          msg_stream_flush();

  // Compilation progress.  Should probably have different name.
  virtual void          report_progress( ProgressReporter::ProgressState );

  // Execution time progress.
  // Percent is number between 0.0-1.0
  virtual void          update_progress(double percent);
  virtual void          update_progress(int current, int max);
  virtual void          increment_progress();

private:
  LogFile log_;
};


} // Namespace SCIRun

#endif
