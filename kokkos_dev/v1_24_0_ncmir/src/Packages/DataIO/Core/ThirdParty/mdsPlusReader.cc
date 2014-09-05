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
  multiple connections (e.g. MDSPlus sockets).

  For more information on the calls see mdsPlusAPI.c
*/

#include <sci_defs/mdsplus_defs.h>

#include <Packages/DataIO/Core/ThirdParty/mdsPlusReader.h>

#include "mdsPlusAPI.h"

namespace DataIO {

using namespace SCIRun;

Mutex MDSPlusReader::mdsPlusLock_( "MDS Plus Mutex Lock" );

MDSPlusReader::MDSPlusReader() : socket_(-1)
{
}

// Simple connection interface that insures thread safe calls and the correct socket.
int
MDSPlusReader::connect( std::string server )
{
#ifdef HAVE_MDSPLUS
  int retVal;

  mdsPlusLock_.lock();

  // Insure that there will not be a disconnect if another connection is present.
  MDS_SetSocket( -1 );

  /* Connect to MDSplus */
  socket_ = MDS_Connect((char*)server.c_str());

  if( socket_ > 0 )
    retVal = 0;
  else
    retVal = -1;

  mdsPlusLock_.unlock();

  return retVal;
#else
  return -1;
#endif
}

// Simple open interface that insures thread safe calls and the correct socket.
int
MDSPlusReader::open( const std::string tree,
		     const unsigned int shot )
{
#ifdef HAVE_MDSPLUS
  int retVal;

  mdsPlusLock_.lock();

  MDS_SetSocket( socket_ );

  /* Open tree */
  retVal = MDS_Open((char*)tree.c_str(), shot);

  mdsPlusLock_.unlock();

  return retVal;
#else
  return -1;
#endif
}

// Simple disconnect interface that insures thread safe calls and
// the correct socket.
void
MDSPlusReader::disconnect()
{
#ifdef HAVE_MDSPLUS
  mdsPlusLock_.lock();

  MDS_SetSocket( socket_ );

  /* Disconnect to MDSplus */
  MDS_Disconnect();

  socket_ = -1;

  mdsPlusLock_.unlock();
#endif
}

// Simple valid interface that insures thread safe calls
// and the correct socket.
bool
MDSPlusReader::valid( const std::string signal )
{
#ifdef HAVE_MDSPLUS
  mdsPlusLock_.lock();

  MDS_SetSocket( socket_ );

  bool valid = is_valid( signal.c_str() );

  mdsPlusLock_.unlock();

  return valid;
#else
  return false;
#endif
}


// Simple type (data type) interface that insures thread safe calls
// and the correct socket.
int
MDSPlusReader::type( const std::string signal )
{
#ifdef HAVE_MDSPLUS
  mdsPlusLock_.lock();

  MDS_SetSocket( socket_ );

  int type = get_type( signal.c_str() );

  mdsPlusLock_.unlock();

  return type;
#else
  return 0;
#endif
}

// Simple rank (number of dimensions) interface that insures thread safe calls
// and the correct socket.
int
MDSPlusReader::rank( const std::string signal )
{
#ifdef HAVE_MDSPLUS
  mdsPlusLock_.lock();

  MDS_SetSocket( socket_ );

  int rank = get_rank( signal.c_str() );

  mdsPlusLock_.unlock();

  return rank;
#else
  return 0;
#endif
}

// Simple dims (number of dimension) interface that insures thread safe calls
// and the correct socket.
int
MDSPlusReader::dims( const std::string signal, unsigned int** dims )
{
#ifdef HAVE_MDSPLUS
  mdsPlusLock_.lock();

  MDS_SetSocket( socket_ );

  int ndims = get_dims( signal.c_str(), dims );

  mdsPlusLock_.unlock();

  return ndims;
#else
  return 0;
#endif
}


// Simple interface that insures thread safe calls and the correct socket.
std::string
MDSPlusReader::name( const unsigned int nid )
{
#ifdef HAVE_MDSPLUS
  mdsPlusLock_.lock();

  MDS_SetSocket( socket_ );

  std::string namestr( get_name(nid) );

  mdsPlusLock_.unlock();

  return namestr;
#else
  return "";
#endif
}


// Simple interface that insures thread safe calls and the correct socket.
unsigned int
MDSPlusReader::names( const std::string signal,
		      std::vector<std::string> &signals,
		      bool recurse, bool absolute, bool type )
{
#ifdef HAVE_MDSPLUS
  mdsPlusLock_.lock();

  MDS_SetSocket( socket_ );

  char *names = NULL;

  unsigned int nnames =
    get_names(signal.c_str(), &names, recurse, absolute, type);

  signals.resize(nnames);

  unsigned int m = 0;

  for( unsigned int n=0; n<nnames; n++ ) {

    std::string name( &(names[m]) );

    signals[n] = name;

    m += name.length() + 1;
  }

  if( names )
    free( names );

  mdsPlusLock_.unlock();

  return nnames;
#else
  return 0;
#endif
}

unsigned int 
MDSPlusReader::nids( const std::string signal, unsigned int **nids )
{
#ifdef HAVE_MDSPLUS
  mdsPlusLock_.lock();

  MDS_SetSocket( socket_ );

  unsigned int nnids = get_nids( signal.c_str(), nids );

  mdsPlusLock_.unlock();

  return nnids;
#else
  return NULL;
#endif
}

void *
MDSPlusReader::values( const std::string signal, unsigned int dtype )
{
#ifdef HAVE_MDSPLUS
  mdsPlusLock_.lock();

  MDS_SetSocket( socket_ );

  void * values = get_values( signal.c_str(), dtype );

  mdsPlusLock_.unlock();

  return values;
#else
  return NULL;
#endif
}

}
