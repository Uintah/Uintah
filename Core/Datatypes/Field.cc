
#include <Core/Datatypes/Field.h>

namespace SCIRun{

// initialize the static member type_id
PersistentTypeID Field::type_id(type_name(0), "FieldData", NULL);


Field::Field(data_location at) :
  data_at_(at)
{
}

Field::~Field()
{
}

const double FIELD_VERSION = 1.0;

void 
Field::io(Piostream& stream){

  stream.begin_class(Field::type_name(0).c_str(), FIELD_VERSION);
  data_location &tmp = data_at_;
  Pio(stream, (unsigned int&)tmp);
  stream.end_class();
}

const string 
Field::type_name(int)
{
  static const string name = "Field";
  return name;
}

}
