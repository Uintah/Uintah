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
 *  CCAException.h: 
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#ifndef SCIRun_Framework_CCAException_h
#define SCIRun_Framework_CCAException_h

#include <Core/Exceptions/Exception.h>
#include <string>

namespace SCIRun {
  class CCAException : public Exception {
  public:
    CCAException(const std::string& description);
    CCAException(const CCAException&);
    virtual ~CCAException();

    virtual const char* message() const;
    virtual const char* type() const;
  private:
    std::string description;

    CCAException& operator=(const CCAException&);
  };
}

#endif
