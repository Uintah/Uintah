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
 *  PartManager.cc
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   Oct 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <iostream>
#include <typeinfo>
#include <Core/Parts/PartManager.h>
#include <vector>

namespace SCIRun {
  
using std::vector;
using std::cerr;
using std::endl;

PartManager::PartManager( PartInterface *parent, const string &name, 
		      bool initialize) : 
  Part( parent, name, this ), 
  PartInterface( this, parent, "PartManager", false )
{
    if ( initialize ) PartInterface::init();
}

PartManager::~PartManager()
{
}

int
PartManager::add_part( Part *part )
{
  int pos = parts_.size();
  parts_.push_back( part );
  if ( pos == 0 )
    current_ = part;

  has_part( part->name() );

  if ( part->interface() )
    part->interface()->set_parent( this );

  return pos;
};

void
PartManager::select_part( int i )
{
  if ( i < int(parts_.size()) ) {
    current_ = parts_[i];
    part_selected( i );
  }
}

// void
// PartManager::report_children( SlotBase1<PartInterface*> &slot )
// {
//   cerr << type_ << " report " << parts_.size() << " parts " << endl;

//   for (unsigned i=0; i<parts_.size(); i++)
//     slot.send(parts_[i]);
// }

} // namespace SCIRun


