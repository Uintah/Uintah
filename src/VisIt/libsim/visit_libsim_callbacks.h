/*
 * The MIT License
 *
 * Copyright (c) 1997-2014 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
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

#ifndef UINTAH_VISIT_LIBSIM_CALLBACKS_H
#define UINTAH_VISIT_LIBSIM_CALLBACKS_H

/**************************************
        
CLASS
   visit_libsim_callbacks
        
   Short description...
        
GENERAL INFORMATION
        
   visit_init
        
   Allen R. Sanderson
   Scientific Computing and Imaging Institute
   University of Utah
        
KEYWORDS
   VisIt, libsim, in-situ
        
DESCRIPTION
   Long description...
        
WARNING
        

****************************************/

namespace Uintah {

  int visit_BroadcastIntCallback(int *value, int sender);
  int visit_BroadcastStringCallback(char *str, int len, int sender);
  void visit_SlaveProcessCallback();

  void visit_ControlCommandCallback(const char *cmd, const char *args, void *cbdata);

  int visit_ProcessVisItCommand( visit_simulation_data *sim );

  void visit_MaxTimeStepCallback (char *val, void *cbdata);
  void visit_MaxTimeCallback     (char *val, void *cbdata);
  void visit_DeltaTCallback      (char *val, void *cbdata);
  void visit_DeltaTMinCallback   (char *val, void *cbdata);
  void visit_DeltaTMaxCallback   (char *val, void *cbdata);
  void visit_DeltaTFactorCallback(char *val, void *cbdata);
  void visit_MaxWallTimeCallback (char *val, void *cbdata);
 
  void visit_UPSVariableTableCallback(char *val, void *cbdata);
  void visit_OutputIntervalVariableTableCallback(char *val, void *cbdata);

  void visit_ImageCallback(int val, void *cbdata);
  void visit_ImageFilenameCallback(char *val, void *cbdata);
  void visit_ImageHeightCallback(char *val, void *cbdata);
  void visit_ImageWidthCallback(char *val, void *cbdata);
  void visit_ImageFormatCallback(int val, void *cbdata);

} // End namespace Uintah

#endif
