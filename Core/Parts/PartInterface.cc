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
 *  Interface.cc
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   Sep 2001
 *
 *  Copyright (C) 2001 SCI Group
 */


#include <Core/Persistent/Pstreams.h>
#include <Core/Parts/PartInterface.h>
#include <Core/Parts/GraphPart.h>


namespace SCIRun {

PartInterface::PartInterface( Part *part, PartInterface *parent, 
			      const string &type )
  : type_(type), part_(part),  parent_(parent)
{
  cerr << "PI = " << part->name() << endl;
  cerr << this << "  part = " << part << endl;
  if ( !dynamic_cast<GraphPart *>(this) )
    cerr <<"could not cast\n";
  else
    cerr <<"ok\n";

  if ( !dynamic_cast<GraphPart *>(part) )
    cerr <<"could not cast part\n";
  else
    cerr <<"part ok\n";
  if ( parent )
    parent->add_child(this);
  
}
 
PartInterface::~PartInterface()
{
  for (unsigned i=0; i<children_.size(); i++)
    delete children_[i];
}


void
PartInterface::add_child( PartInterface *child )
{
  children_.push_back(child);

  cerr << "PartInterface emit has_child\n";
  // signal you have a new child
  has_child( child  );
}


void
PartInterface::rem_child( PartInterface *child )
{
  vector<PartInterface *>::iterator i;
  for (i=children_.begin(); i!=children_.end(); i++)
    if ( *i == child ) {
      children_.erase( i );
      return;
    }
}

} // namespace SCIRun

