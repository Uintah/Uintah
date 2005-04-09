
#include <Datatypes/cMatrix.h>
#include <iostream.h>

PersistentTypeID cMatrix::type_id("cMatrix", "Datatype", 0);

void cMatrix::io(Piostream&) {
  cerr << "cMatrix::io not finished\n";
}
