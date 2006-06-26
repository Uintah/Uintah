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
 *  GroupMutex: Standard locking primitive for groups of threads
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

namespace SCIRun {

class Mutex_private;

/**************************************

 CLASS
 GroupMutex

 KEYWORDS
 Thread

 DESCRIPTION
 Provides a simple <b>Mut</b>ual <b>Ex</b>clusion primitive for group of threads.
 Atomic <b>lock(gid)</b> and <b>unlock()</b> will lock (group with id=gid gains the lock)
 and unlock the mutex. This is not a recursive Mutex (See <b>RecursiveMutex</b>), and calling
 lock() in a nested call will result in an error or deadlock.

****************************************/
class GroupMutex {
public:
  //////////
  // Create the mutex.  The mutex is allocated in the unlocked
  // state. <i>name</i> should be a static string which describes
  // the primitive for debugging purposes.  
  GroupMutex(const char* name);

  //////////
  // Destroy the mutex.  Destroying the mutex in the locked state
  // has undefined results.
  ~GroupMutex();

  //////////
  // Acquire the Mutex.  This method will block until the group gid gains 
  // the mutex. 
  void lock(int gid);


  //////////
  // Release the Mutex, unblocking any other threads that are
  // blocked waiting for the Mutex.
  void unlock();
private:
  bool locked;
  bool gid;
  const char* name_;

  // Cannot copy them
  GroupMutex(const GroupMutex&);
  GroupMutex& operator=(const GroupMutex&);
};

} // End namespace SCIRun

#endif


