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
//    File   : Colormap2.h
//    Author : Milan Ikits
//    Date   : Mon Jul  5 18:33:12 2004

#ifndef Colormap2_h
#define Colormap2_h

#include <Core/Datatypes/Datatype.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Thread/Mutex.h>
#include <vector>

namespace Volume {

class CM2Widget;

using std::vector;

class Colormap2 : public SCIRun::Datatype
{
public:
  Colormap2();
  Colormap2(vector<CM2Widget*>& widgets, bool updating, bool faux);
  virtual ~Colormap2();

  inline vector<CM2Widget*>& widgets() { return widgets_; }
  
  inline bool updating() { return updating_; }
  inline bool faux() { return faux_; }

  virtual void io(SCIRun::Piostream&);
  static SCIRun::PersistentTypeID type_id;

protected:
  bool updating_;
  bool faux_;
  vector<CM2Widget *> widgets_;
};

typedef SCIRun::LockingHandle<Colormap2> Colormap2Handle;

} // End namespace Volume

#endif // Colormap2_h
