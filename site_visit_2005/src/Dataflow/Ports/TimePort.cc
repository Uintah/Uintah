//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2004 Scientific Computing and Imaging Institute,
//  University of Utah.
//  
//  License for the specific language governing rights and limitations under
//  Permission is hereby granted, free of charge, to any person obtaining a
//  copy of this software and associated documentation files (the "Software"),
//  to deal in the Software without restriction, including without limitation
//  the rights to use, copy, modify, merge, publish, distribute, sublicense,
//  and/or sell copies of the Software, and to permit persons to whom the
//  Software is furnished to do so, subject to the following conditions:
//  
//  The above copyright notice and this permission notice shall be included
//  in all copies or substantial portions of the Software.
//  
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//  DEALINGS IN THE SOFTWARE.
//  
//    File   : TimePort.cc
//    Author : Martin Cole
//    Date   : Thu Apr 14 12:48:19 2005


#include <Dataflow/Ports/TimePort.h>
#include <Dataflow/Network/Module.h>

namespace SCIRun {

extern "C" {
  IPort* make_TimeIPort(Module* module, const string& name) {
    return scinew TimeIPort(module, name);
  }
  OPort* make_TimeOPort(Module* module, const string& name) {
    return scinew TimeOPort(module, name);
  }
}

template<> string SimpleIPort<TimeViewerHandle>::port_type_("Time");
template<> string SimpleIPort<TimeViewerHandle>::port_color_("SpringGreen2");


TimeViewer::TimeViewer() :
  ref_cnt(0),
  lock("TimeViewer ref_cnt lock")
{}

TimeViewer::~TimeViewer()
{}


TimeData::TimeData() :
  TimeViewer(),
  crowd_(""),
  started_(0.0L),
  now_(0.0L),
  last_(0.0L)
{}

TimeData::~TimeData() 
{}

double
TimeData::view_started() 
{
  double rval;
  crowd_.readLock();
  rval = started_;
  crowd_.readUnlock();
  return rval;
}

double
TimeData::view_elapsed_since_start() 
{
  double rval;
  crowd_.readLock();
  rval = now_ - started_;
  crowd_.readUnlock();
  return rval;
}

double
TimeData::view_elapsed_since_last() 
{
  double rval;
  crowd_.readLock();
  rval = now_ - last_;
  crowd_.readUnlock();
  return rval;
}

void
TimeData::set_started(double t) 
{
  crowd_.writeLock();
  started_ = t;
  crowd_.writeUnlock();
}

void
TimeData::set_now(double t) 
{
  crowd_.writeLock();
  last_ = now_;
  now_  = t;
  crowd_.writeUnlock();
}



} // End namespace SCIRun
