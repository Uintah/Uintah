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
 *  GraphPart.cc
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   Sep 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <iostream>
#include <typeinfo>
#include <Core/Parts/GraphPart.h>
#include <vector>
#include <Core/2d/LockedPolyline.h>

namespace SCIRun {
  
using std::vector;
using std::cerr;
using std::endl;

GraphPart::GraphPart( PartInterface *parent, const string &name, 
		      bool initialize) : 
  Part( parent, name, this ), 
  PartInterface( this, parent, "GraphGui", false )
{
    if ( initialize ) PartInterface::init();
}

GraphPart::~GraphPart()
{
}

void 
GraphPart::set_num_lines( int n )
{
  data_.resize(n);
  reset(n);
  
}
  
#ifdef CHRIS
void 
GraphPart::add_values( unsigned item, const vector<double> &v)
{
  if (item >= data_.size()) {
    cerr << "add_values item " << item << " >= data_ size " 
	 << data_.size() << endl;
    return;
  }

  for (unsigned i=0; i<v.size(); ++i) 
    data_[item].push_back(v[i]);
  new_values(item,v);
}
#else
void 
GraphPart::add_values( const vector<double> &v)
{
  if (v.size() != data_.size()) {
    cerr << "add_values size " << v.size() << " != data_ size " 
	 << data_.size() << endl;
    return;
  }

  for (unsigned i=0; i<v.size(); i++)
    data_[i].push_back(v[i]);
  new_values( v );
};
#endif

} // namespace SCIRun


