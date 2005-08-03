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
//    File   : TimePort.h
//    Author : Martin Cole
//    Date   : Wed Apr 13 14:42:23 2005

#if !defined(TimePort_h)
#define TimePort_h

#include <Dataflow/Ports/SimplePort.h>

namespace SCIRun {

class TimeViewer 
{
public:

  TimeViewer();
  virtual ~TimeViewer();
  //! interface to time vars.
  virtual double view_started() = 0;
  virtual double view_elapsed_since_start() = 0;
  virtual double view_elapsed_since_last() = 0;

  //! needed for LockingHandle<T>
  int      ref_cnt;
  Mutex    lock;
};

class TimeData : public TimeViewer
{
public:
  TimeData();
  virtual ~TimeData();

  virtual double view_started();
  virtual double view_elapsed_since_start();
  virtual double view_elapsed_since_last();

  void set_started(double);
  void set_now(double);

private:
  CrowdMonitor     crowd_;

  double           started_;
  double           now_;
  double           last_;
};

typedef LockingHandle<TimeViewer> TimeViewerHandle;
typedef SimpleIPort<TimeViewerHandle> TimeIPort;
typedef SimpleOPort<TimeViewerHandle> TimeOPort;

}

#endif // TimePort_h
