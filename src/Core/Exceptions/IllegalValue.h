/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/



/*
 *  IllegalValue.h: Generic exception for invalid values
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#ifndef Core_Exceptions_IllegalValue_h
#define Core_Exceptions_IllegalValue_h

#include <Core/Exceptions/Exception.h>
#include <sgi_stl_warnings_off.h>
#include <string>
#include <sstream>
#include <sgi_stl_warnings_on.h>

namespace SCIRun {
  template <class T>
  class IllegalValue : public Exception {
  public:
    IllegalValue(const std::string&, const T& value);
    IllegalValue(const IllegalValue&);
    virtual ~IllegalValue();
    virtual const char* message() const;
    virtual const char* type() const;
  protected:
  private:
    std::string message_;
    IllegalValue& operator=(const IllegalValue&);
  };

  template <class T>
  IllegalValue<T>::IllegalValue(const std::string& message, const T& value)
  {
    std::ostringstream msgbuf;
    msgbuf << message << ", value=" << value;
    message_=msgbuf.str();
  }

  template <class T>
  IllegalValue<T>::IllegalValue(const IllegalValue& copy)
    : message_(copy.message_)
  {
  }

  template <class T>
  IllegalValue<T>::~IllegalValue()
  {
  }

  template <class T>
  const char* IllegalValue<T>::message() const
  {
    return message_.c_str();
  }

  template <class T>
  const char* IllegalValue<T>::type() const
  {
    return "IllegalValue";
  }

} // End namespace SCIRun

#endif


