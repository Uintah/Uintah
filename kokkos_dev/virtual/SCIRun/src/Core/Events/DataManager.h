//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2004 Scientific Computing and Imaging Institute,
//  University of Utah.
//  
//  
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
//    File   : DataManager.h
//    Author : Martin Cole
//    Date   : Tue Sep 12 09:55:13 2006


#if !defined(DataManager_h)
#define DataManager_h

#include <Core/Util/ThrottledRunnable.h>
#include <Core/Thread/Thread.h>
#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/Matrix.h>
#include <Core/Datatypes/NrrdData.h>
#include <Core/Events/Tools/ToolManager.h>
#include <Core/Events/EventManager.h>

namespace SCIRun {

class RenderParams;

class SCISHARE DataManager : public ThrottledRunnable 
{
public:
  DataManager();
  virtual ~DataManager();

  virtual bool          iterate();

  FieldHandle  get_field(unsigned int id);
  MatrixHandle get_matrix(unsigned int id);
  NrrdDataHandle   get_nrrd(unsigned int id);

  unsigned int load_field(string fname);
  unsigned int load_matrix(string fname);
  unsigned int load_nrrd(string fname);

  // sends Scene Graph event with the rendered geometry.
  bool           show_field(unsigned int fld_id);
  bool           toggle_field_visibility(unsigned int fld_id);
  void           selection_target_changed(unsigned int fid);
  unsigned int   get_selection_target() { return sel_fid_; }

private:
  void           set_render_params(unsigned int);

  Mutex                               lock_;
  
  map<unsigned int, NrrdDataHandle>   nrrds_;
  map<unsigned int, MatrixHandle>     mats_;
  map<unsigned int, FieldHandle>      fields_;
  
  unsigned int                        sel_fid_;
  RenderParams                       *params_;

  ToolManager                         tm_;
  EventManager::event_mailbox_t      *events_;

  static unsigned int                 next_id_;

};

}

#endif //DataManager_h
