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
