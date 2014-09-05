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
 *  PartManager.h
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   Oct 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#ifndef SCI_PartManager_h
#define SCI_PartManager_h 

#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>
#include <Core/Util/Signals.h>
#include <Core/Parts/Part.h>
#include <Core/Parts/PartInterface.h>

namespace SCIRun {
  
class PartManager : public Part, public PartInterface  {
private:
  vector< Part *> parts_;
  Part *current_;

public:
  PartManager( PartInterface *parent = 0, const string &name="PartManager",
	       bool=true);
  virtual ~PartManager();

  virtual int add_part( Part * );

  Part *current() { return current_;}
  Part *operator ->() { return current_; }

//   virtual void report_children( SlotBase1<PartInterface*> &slot );

  // Slot
  void select_part( int );

  // Signal
  Signal1< const string & > has_part;
  Signal1< int > part_selected;
};


template<class T>
class PartManagerOf : public PartManager {
public:
  PartManagerOf( PartInterface *parent = 0, 
		 const string &name="PartManagerOf",
		 bool init=true) 
    : PartManager( parent, name, init )
  {
  }

  virtual ~PartManagerOf() {}

  virtual int add_part( T *part ) { return PartManager::add_part(part); }

  T *operator ->() { return static_cast<T*>(current()); }
};

} // namespace SCIRun

#endif // SCI_PartManager_h

