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
 *  CleanupManager.cc: Manage exitAll callbacks.
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   November 2004
 *
 *  Copyright (C) 2004 SCI Group
 */

//  How to use this class:
//  
//  #include <Core/Thread/CleanupManager.h>
//  
//  Make a callback function that takes zero arguements and returns
//  void.  Be very careful about introducing crashes into this
//  callback function as it then becomes difficult to exit from
//  scirun.  It's recommended that you avoid calling scinew from
//  within your callback.
//  
//  Register with CleanupManager::add_callback(YOUR_CALLBACK_HERE);
//  
//  Your callback will only ever be called once, no matter how many
//  times you register it.  In addition you can unregister it or
//  design it such that it doesn't do anything if it doesn't need to.

#ifndef SCI_project_CleanupManager_h
#define SCI_project_CleanupManager_h 1

#include <Core/share/share.h>
#include <Core/Thread/Mutex.h>
#include <vector>

namespace SCIRun {

typedef void (*CleanupManagerCallback)();

class SCICORESHARE CleanupManager {
public:
  
  static void add_callback(CleanupManagerCallback callback);
  static void remove_callback(CleanupManagerCallback callback);

  static void call_callbacks();

protected:
  static std::vector<CleanupManagerCallback> callbacks_;
};

} // End namespace SCIRun


#endif /* SCI_project_CleanupManager_h */
