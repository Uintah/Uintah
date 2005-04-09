
/*
 *  SurfaceRegister.cc:
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   Jul 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

static RegisterModule db1("Surfaces", "SurfaceReader", make_SurfaceReader);
static RegisterModule db2("Readers", "SurfaceReader", make_SurfaceReader);

