
#ifndef UDA2NRRD_BC_FUNCTIONS_H
#define UDA2NRRD_BC_FUNCTIONS_H

#include <Core/Geometry/IntVector.h>

bool is_periodic_bcs( SCIRun::IntVector cellir,
                      SCIRun::IntVector ir );

void get_periodic_bcs_range( SCIRun::IntVector   cellmax, 
                             SCIRun::IntVector   datamax,
                             SCIRun::IntVector   range,
                             SCIRun::IntVector & newrange);

#endif // UDA2NRRD_BC_FUNCTIONS_H
