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
 *  MDSPlusReader.cc:
 *
 *  Written by:
 *   Allen Sanderson
 *   School of Computing
 *   University of Utah
 *   March 2002
 *
 *  Copyright (C) 2002 SCI Group
 */

/*
  This is a C++ interface for fetching data from a MDSPlus Server.

  Unfortunately, because MDSPlus has several short commings this interface
  is need is to provide a link between SCIRun and MDSPlus. The two items
  addressed are making MDSPlus thread safe and allowing seemless access to
  multiple servers and, or trees.
*/

#include <sci_defs.h>

#include <Core/Thread/Mutex.h>

#include <string>

namespace Fusion {

using namespace SCIRun;

class MDSPlusReader {
public:
  MDSPlusReader();

  int connect( const std::string server );
  int open( const std::string tree, int shot );
  void disconnect();
  int rank( const std::string signal );
  int size( const std::string signal );
  double *grid( const std::string axis, int *dims );

  int slice_ids( int **nids );
  std::string slice_name( const int *nids, int slice );
  double slice_time( const std::string name );

  double *slice_data( const std::string name,
		      const std::string space,
		      const std::string node,
		      int *dims );

protected:
  int socket_;

  static Mutex mdsPlusLock_;
};
}
