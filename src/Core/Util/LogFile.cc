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

// LogFile.cc


#include <Core/Util/LogFile.h>

namespace SCIRun {

LogFile::LogFile(std::string filename) :
  lock("servicelog lock"), 
  ref_cnt(0),
  haslog_(false) 
{	
  if (filename != "") 
  {
    try 
    {
      logfile_.open(filename.c_str());
      haslog_ = true;
    }
    catch(...) 
    {
      haslog_ = false;
    }
  }
}	

void LogFile::putmsg(std::string msg)
{
  if (haslog_) 
  {
    try 
    {
      lock.lock();
      logfile_ << msg << std::endl;
      logfile_.flush();
      lock.unlock();
    }
    catch(...) 
    {	// make sure we always unlock
      lock.unlock();
      throw;
    }
  }
}	

}
