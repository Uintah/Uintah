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


#include <Core/Datatypes/Field.h>

namespace SCIRun{

// initialize the static member type_id
PersistentTypeID Field::type_id("Field", "PropertyManager", 0);


Field::Field(data_location at) :
  data_at_(at)
{
}

Field::~Field()
{
}

const int FIELD_VERSION = 1;

void 
Field::io(Piostream& stream){

  stream.begin_class("Field", FIELD_VERSION);
  data_location &tmp = data_at_;
  Pio(stream, (unsigned int&)tmp);
  PropertyManager::io(stream);
  stream.end_class();
}

}
