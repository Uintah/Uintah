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
 *  ProcessInfo.h:
 *
 *  Written by:
 *   Author: Randy Jones
 *   Department of Computer Science
 *   University of Utah
 *   Date: 2004/02/05
 *
 *  Copyright (C) 2004 SCI Group
 */

#ifndef Core_OS_ProcessInfo_h
#define Core_OS_ProcessInfo_h 1


namespace SCIRun {


  class ProcessInfo {

  public:

    enum /*info_type*/ {
      MEM_SIZE,
      MEM_RSS
    };

    static bool          IsSupported       ( int info_type );
    static unsigned long GetInfo           ( int info_type );

    static unsigned long GetMemoryUsed     ( void ) { return GetInfo( MEM_SIZE ); }
    static unsigned long GetMemoryResident ( void ) { return GetInfo( MEM_RSS  ); }

  private:

    ProcessInfo::ProcessInfo  ( void ) {}
    ProcessInfo::~ProcessInfo ( void ) {}

    static unsigned long LinuxGetInfo ( int info_type );

  }; // class ProcessInfo {


} // namespace SCIRun {


#endif // #ifndef Core_OS_ProcessInfo_h
