
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

static RegisterModule db1("Surfaces", "SurfaceWriter", make_SurfaceWriter);
static RegisterModule db2("Writers", "SurfaceWriter", make_SurfaceWriter);

