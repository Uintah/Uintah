#include <Packages/Uintah/Core/Datatypes/TensorField.h>

#include <Core/Containers/String.h>

#include <iostream>

using std::cerr;

namespace Uintah {

PersistentTypeID TensorField::type_id("TensorField", "Datatype", 0);

#define TENSORFIELD_VERSION 1

void TensorField::io(Piostream& stream)
{
    /*int version=*/stream.begin_class("TensorField", TENSORFIELD_VERSION);
    stream.end_class();
}

void TensorField::get_bounds(Point& min, Point& max){
    if(!have_bounds){
	compute_bounds();
	have_bounds=1;
	diagonal=bmax-bmin;
    }
    max=bmax;
    min=bmin;
}

} // End namespace Uintah
