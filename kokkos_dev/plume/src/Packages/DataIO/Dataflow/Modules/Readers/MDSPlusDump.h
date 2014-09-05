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


//    File   : MDSPlusDump.h
//    Author : Allen Sanderson
//             School of Computing
//             University of Utah
//    Date   : July 2004

#ifndef MDSPLUS_DUMP_API
#define MDSPLUS_DUMP_API

#include <sci_defs/mdsplus_defs.h>

#include <Packages/DataIO/Core/ThirdParty/mdsPlusReader.h>

namespace DataIO {

using namespace std;

class MDSPlusDump {
public:
  MDSPlusDump( ostream *iostr );

  void tab( );

  int tree( std::string server,
	    std::string tree,
	    unsigned int shot,
	    std::string start_node,
	    unsigned int max_nodes );

  int nodes  ( const std::string node );
  int node   ( const std::string path, const std::string node );
  int signals( const std::string signal );
  int signal ( const std::string path, const std::string node );
  int labels ( const std::string signal );
  int label  ( const std::string path, const std::string node );

  int dataspace( const std::string signal );
  int datatype ( const std::string signal );

  std::string error() { return error_msg_; };

private:
  unsigned int indent_;
  unsigned int max_indent_;

  ostream *iostr_;
  std::string error_msg_;

  MDSPlusReader mds_;
};
}

#endif  // MDSPLUS_DUMP_API
