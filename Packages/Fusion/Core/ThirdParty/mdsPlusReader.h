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

#define DTYPE_UCHAR   2
#define DTYPE_USHORT  3
#define DTYPE_ULONG   4
#define DTYPE_ULONGLONG 5
#define DTYPE_CHAR    6
#define DTYPE_SHORT   7
#define DTYPE_LONG    8
#define DTYPE_LONGLONG 9
#define DTYPE_FLOAT   10
#define DTYPE_DOUBLE  11
#define DTYPE_COMPLEX 12
#define DTYPE_COMPLEX_DOUBLE 13
#define DTYPE_CSTRING 14
#define DTYPE_FS  52
#define DTYPE_FT  53
#define DTYPE_FSC 54
#define DTYPE_FTC 55

namespace Fusion {

using namespace SCIRun;

class MDSPlusReader {
public:
  MDSPlusReader();

  int connect( const std::string server );
  int open( const std::string tree, int shot );
  void disconnect();

  bool valid( const std::string signal );

  int type( const std::string signal );
  int rank( const std::string signal );
  int dims( const std::string signal, int** dims );
  void* values( const std::string signal, int dtype );


  double *grid( const std::string axis, int **dims  );
  int slice_ids( int **nids );
  std::string slice_name( const int nid );
  double slice_time( const std::string name );

  double *slice_data( const std::string name,
		      const std::string space,
		      const std::string node,
		      int **dims );

protected:
  int socket_;

  static Mutex mdsPlusLock_;
};
}
