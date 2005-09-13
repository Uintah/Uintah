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
 *  Part.h
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   Sep 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#ifndef SCI_Part_h
#define SCI_Part_h 

#include <sgi_stl_warnings_off.h>
#include <string>
#include <map>
#include <vector>
#include <sgi_stl_warnings_on.h>
#include <stdio.h>

namespace SCIRun {

using std::string;
using std::vector;
using std::map;
using std::pair;
  
class PartInterface;

typedef vector<map<string,vector<unsigned char> > > property_vector;
typedef map<string,vector<unsigned char> >::iterator pv_iter;

class Part {
protected:
  PartInterface *parent_interface_;
  PartInterface *interface_;

  string name_;
  property_vector properties_;

public:
  Part( PartInterface *parent=0, const string name="", PartInterface *i=0 ) 
    : parent_interface_(parent), interface_(i), name_(name) 
  { 
  }
  virtual ~Part() {/* if (interface_) delete interface_;*/ }

  PartInterface *interface() { return interface_; }
  string name() { return name_; }

  void set_property(int id, string name, vector<unsigned char> data) 
  { 
    if ((int)properties_.size()<=id)
	properties_.resize(id+1);
    properties_[id].insert(pair<string, vector<unsigned char> >(name,data));
  }

  //! data[0] is the valid bool: set to 1 if this call succeeds, 0 otherwise
  void get_property(int id, string name , vector<unsigned char>& data) 
  {
    if ((int)properties_.size()>id) {
      pv_iter i = properties_[id].find(name);
      if (i != properties_[id].end()) {
        int length=data.size();
        data[0]=1;
        // copy from property to get buffer (data)
	for (int loop=1;loop<length;++loop)
          data[loop]=(*i).second[loop-1];
      } else { data[0]=0; }
    } else { data[0]=0; }
  }
};

} // namespace SCIRun

#endif // SCI_Part_h
