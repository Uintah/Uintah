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
 *  Reference.h: A serializable "pointer" to an object
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#ifndef CCA_PIDL_Reference_h
#define CCA_PIDL_Reference_h

#include <Core/CCA/Comm/SpChannel.h>

namespace SCIRun {
/**************************************
 
CLASS
   Reference
   
KEYWORDS
   Reference, PIDL
   
DESCRIPTION
   A remote reference.  This class is internal to PIDL and should not
   be used outside of PIDL or sidl generated code.  It contains a 
   spchannel and the vtable base offset.
****************************************/
  class Reference {
  public:
    //////////
    // Empty constructor.  Initalizes the channel to nil
    Reference();

    //////////
    // Constructor which accepts a channel.
    Reference(SpChannel* n_chan);

    //////////
    // Copy the reference. 
    Reference(const Reference&);

    void cloneTo(Reference &Clone);

    //////////
    // Clone the reference, duplicate everything
    Reference *  clone();

    //////////
    // Copy the reference.  
    Reference& operator=(const Reference&);

    //////////
    // Destructor. 
    ~Reference();

    //////////
    // Return the vtable base
    int getVtableBase() const;

    //////////
    // Proxy's communication class. 
    SpChannel* chan;

    //////////
    // The vtable base offset
    int d_vtable_base;
  private:
    // primary==false means chan comes from SPFactory(false)
    // thus should not be deleted in the destructor.
    bool primary; 

    //////////
    // Copy the reference.  
    Reference& _copy(const Reference&);

  };
} // End namespace SCIRun

#endif









