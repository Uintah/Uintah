/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
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
 * HEADER (H) FILE : NetConnection.h
 *
 * DESCRIPTION     : The NetConnection object contains all information 
 *                   associated with a single socket connection.
 *                   This is a wrapper class for the C struct in 
 *                   NetConnectionC.h[c].  This functions as a "READ ONLY" 
 *                   object in that the object cannot be modified directly.
 *                   The only way to modify a NetConnection object is via 
 *                   the NetInterface object.
 *                  
 * 
 * AUTHOR(S)       : Jenny Simpson
 *                   SCI Institute
 *                   University of Utah
 *                 
 * CREATED         : Mon Apr 12 13:28:20 MDT 2004
 * MODIFIED        : Wed May 26 11:57:46 MDT 2004
 * DOCUMENTATION   :
 * NOTES           : 
 *
 * Copyright (C) 2003 SCI Group
*/

#ifndef SCI_Core_Util_NetConnection_h
#define SCI_Core_Util_NetConnection_h

#include <sci_defs/scisock_defs.h>

#ifdef HAVE_SCISOCK


// SCIRun includes
#include <Core/Malloc/Allocator.h>
//#include <SCISockInterface/lib/NetConnectionC.h>
#include <NetConnectionC.h>

// Standard lib includes
#include <iostream>
#include <cassert>
#include <limits>

namespace SCIRun {

// ****************************************************************************
// ************************* Class: NetConnection *****************************
// ****************************************************************************

class NetConnection
{

public:

  //! Constructors
  NetConnection();
  NetConnection( struct NetConnectionC * conn );

  //! Copy constructor
  NetConnection( const NetConnection& nl );

  //! Destructor
  ~NetConnection();

  //! Utility functions
  bool is_open(); 
  int get_type();
  int get_sockfd();
  std::string get_host_name();
  int get_port();
  void print();

  struct NetConnectionC * get_connection();

private:

  //! Member variables

  // Underlying connection object
  struct NetConnectionC * connection_;

};

} // End namespace SCIRun
 
#endif
#endif // SCI_Core_Util_NetConnection_h



