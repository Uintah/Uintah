#include "TensorField.h"

#include <SCICore/Containers/String.h>
#include <iostream>
using std::cerr;

namespace SCICore {
namespace Datatypes {

PersistentTypeID TensorField::type_id("TensorField", "Datatype", 0);

#define TENSORFIELD_VERSION 1

void TensorField::io(Piostream& stream)
{
    /*int version=*/stream.begin_class("TensorField", TENSORFIELD_VERSION);
    stream.end_class();
}

void TensorField::get_bounds(Point&, Point&){
  cerr<<"Get_bounds\n";
}
} // End namespace Datatypes
} // End namespace SCICore
