
/*
 *  VectorFieldRegister.cc:
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   Jul 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

static RegisterModule db1("Fields", "VectorFieldReader", make_VectorFieldReader);
static RegisterModule db2("Readers", "VectorFieldReader", make_VectorFieldReader);

