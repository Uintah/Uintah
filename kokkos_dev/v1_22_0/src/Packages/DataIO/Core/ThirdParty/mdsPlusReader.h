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
#include <vector>

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

namespace DataIO {

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

  int search( const std::string signal,
	      const int regexp,
	      std::vector<std::string> &signals );

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
