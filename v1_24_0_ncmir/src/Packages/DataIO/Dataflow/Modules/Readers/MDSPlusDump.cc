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
 *  MDSPlusDump.c:
 *
 *  Written by:
 *   Allen Sanderson
 *   School of Computing
 *   University of Utah
 *   July 2003
 *
 *  Copyright (C) 2002 SCI Group
 */

#include "MDSPlusDump.h"

namespace DataIO {

using namespace std;

MDSPlusDump::MDSPlusDump( ostream *iostr ) :
  indent_(0)
{
  iostr_ = iostr;
}


void
MDSPlusDump::tab() {

  for( unsigned int i=0; i<indent_; i++ )
    *iostr_ << "   ";
}



int
MDSPlusDump::tree( std::string server,
		   std::string tree,
		   unsigned int shot,
		   std::string start_node,
		   unsigned int max_nodes ) {

  int trys = 0;
  int retVal = 0;

  char shotstr[12];

  sprintf( shotstr, "%d", shot );

  indent_ = 0;
  max_indent_ = max_nodes;

  while( trys < 10 && (retVal = mds_.connect( server.c_str()) ) == -2 ) {
    error_msg_ = "Waiting for the connection to become free.";
//    sleep( 1 );
    trys++;
  }

  /* Connect to MDSplus */
  if( retVal == -2 ) {
    error_msg_ = "Connection to Mds Server " + server + " too busy ... giving up.";
    return -1;
  }
  else if( retVal < 0 ) {
    error_msg_ = "Can not connect to Mds Server " + server;
    return -1;
  }

  // Open tree
  trys = 0;

  while( trys < 10 && (retVal = mds_.open( tree.c_str(), shot) ) == -2 ) {
    error_msg_ = "Waiting for the tree and shot to become free.";
//    sleep( 1 );
    trys++;
  }

  if( retVal == -2 ) {
    error_msg_ = "Opening " + tree + " tree and shot " + shotstr + " too busy ... giving up.";;
    return -1;
  }
  if( retVal < 0 ) {
    error_msg_ = "Can not open " + tree + " tree and shot " + shotstr;
    return -1;
  }

  if( start_node.length() == 0 )
    start_node = "\\" + tree + "::TOP";

  *iostr_ << "MDSPlus \"" << server << "-" << tree << "-" << shot << "\" {" << endl;

  int status;

  status = node( "", start_node );

  *iostr_ << "}" << endl;

  mds_.disconnect();

  return status;
}


int
MDSPlusDump::nodes( const std::string path ) {

  unsigned int *nids = NULL;
  int nnodes = mds_.nids(path.c_str(), &nids);

  for( int i=0; i<nnodes; i++ ) {
    
    string child_name( mds_.name( nids[i] ) );
    
    node( path, child_name );
  }

  if( nids )
    free( nids );

  return 0;
}


int
MDSPlusDump::node( const std::string path, const std::string node ) {

  int status = 0;

  tab();
  *iostr_ << "NODE \"" << node << "\" {" << endl;
  indent_++;

  std::string fullpath;

  if( path.length() )
    fullpath =  path + ".";

  fullpath += node;

  if( !max_indent_ || indent_ <= max_indent_ ) {
    status = nodes  ( fullpath );
    status = signals( fullpath );
    status = labels ( fullpath );
  }

  indent_--;
  tab();
  *iostr_ << "}" << endl;

  return status;
}


int
MDSPlusDump::signals( const std::string path ) {

  vector < string > names;

  int nsignals = mds_.names(path, names, 0, 0, 0);

  for( int n=0; n<nsignals; n++ ) {

    string full_name = path + "." + names[n];

    vector < string > children_signal_names;

    int nchildnames = mds_.names(full_name, children_signal_names, 0, 0, 0);

    if( mds_.valid( full_name.c_str() ) ) {
      signal( path, names[n] );
    } else if( nchildnames > 1 ) {
      node( path, names[n] );
    }
  }

  return 0;
}

int
MDSPlusDump::signal( const std::string path, const std::string node ) {
  int status = 0;

  tab();
  *iostr_ << "SIGNAL \"" << node << "\" {" << endl;
  indent_++;

  std::string fullpath;

  if( path.length() )
    fullpath =  path + ".";

  fullpath += node;

  status = datatype ( fullpath );
  status = dataspace( fullpath );
  status = labels   ( fullpath );

  indent_--;
  tab();
  *iostr_ << "}" << endl;

  return status;
}


int
MDSPlusDump::labels( const std::string path ) {

  int status = 0;

  vector < string > labels;

  int nlabels = mds_.names(path, labels, 0, 0, 1);

  for( int n=0; n<nlabels; n++ )
    status = label( path, labels[n] );
 
  return status;
}

int
MDSPlusDump::label( const std::string path, const std::string node ) {

  return 0;

  int status = 0;

  tab();
  *iostr_ << "LABEL \"" << node << "\" {" << endl;
  indent_++;

  std::string fullpath;

  if( path.length() )
    fullpath =  path + ".";

  fullpath += node;

    //      *iostr_ << mds_.value( full_name.c_str(), DTYPE_CSTRING ) << endl;;

  indent_--;
  tab();
  *iostr_ << "}" << endl;

  return status;
}


int
MDSPlusDump::datatype( const std::string path ) {

  tab();
  
  // Get the mds data type from the signal
  int mds_data_type = mds_.type( path.c_str() );

  switch (mds_data_type) {
  case DTYPE_UCHAR:
  case DTYPE_USHORT:
  case DTYPE_ULONG:
  case DTYPE_ULONGLONG:
  case DTYPE_CHAR:
  case DTYPE_SHORT:
  case DTYPE_LONG:
  case DTYPE_LONGLONG:
    *iostr_ << "DATATYPE \"Int\"" << endl;
    break;

  case DTYPE_FLOAT:
  case DTYPE_FS:
    *iostr_ << "DATATYPE \"Float\"" << endl;
    break;

  case DTYPE_DOUBLE:
  case DTYPE_FT:
    *iostr_ << "DATATYPE \"Double\"" << endl;
    break;
  case DTYPE_CSTRING:
    *iostr_ << "DATATYPE \"String\"" << endl;
    break;
  default:
    *iostr_ << "DATATYPE \"Unknown " << mds_data_type << "\"" << endl;
  }

  return 0;
}


int
MDSPlusDump::dataspace( const std::string path ) {

  unsigned int *dims = NULL; 
  int rank = mds_.dims( path.c_str(), &dims );

  if (rank == 0) {
    /* scalar dataspace */
    
    tab();
    *iostr_ << "DATASPACE  SCALAR { ( 1 ) }" << endl;
  } else if( rank > 0 ) {
    
    tab();
    *iostr_ << "DATASPACE  SIMPLE { ( " << dims[0];

    for( int i = 1; i < rank; i++ )
      *iostr_ << ", " << dims[i];
    
    *iostr_ << " ) }" << endl;

    free( dims );
  }

  return 0;
}



} // End namespace SCIDataIO

