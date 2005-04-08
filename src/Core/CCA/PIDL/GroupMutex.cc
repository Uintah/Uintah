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


/*
 *  Mutex: Standard locking primitive for groups of threads
 *
 *  Written by:
 *   Author: Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   Date: March 2004
 *
 *  Copyright (C) 1997 SCI Group
 */

//TODO: this class will be go to src/Core/Thread eventually
#ifndef Core_Thread_Group_Mutex_h
#define Core_Thread_Group_Mutex_h

using namespace SCIRun;

GroupMutex::GroupMutex(const char* name){
  name_=name;
}

GroupMutex::~GroupMutex(){

}

void
GroupMutex::lock(int gid){
  mutex.lock();
  if(!locked){
    this->gid=gid;
    locked=true;
  }
  mutex.unlock();
  
  if(gid==this->gid) return;
  else
}

void GroupMutex::unlock(){

}

private:
  Mutex_private* priv_;
  const char* name_;

  // Cannot copy them
  Mutex(const Mutex&);
  Mutex& operator=(const Mutex&);
};

} // End namespace SCIRun

#endif


