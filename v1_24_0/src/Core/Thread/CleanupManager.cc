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

#include <Core/Thread/CleanupManager.h>
#include <algorithm>


namespace SCIRun {

std::vector<CleanupManagerCallback> CleanupManager::callbacks_;

void
CleanupManager::add_callback(CleanupManagerCallback callback)
{
  remove_callback(callback);
  callbacks_.push_back(callback);
}

void
CleanupManager::remove_callback(CleanupManagerCallback callback)
{
  std::remove(callbacks_.begin(), callbacks_.end(), callback);
}

void
CleanupManager::call_callbacks()
{
  for (unsigned int i = 0; i < callbacks_.size(); i++)
  {
    callbacks_[i]();
  }
  callbacks_.clear();
}


} // End namespace SCIRun

