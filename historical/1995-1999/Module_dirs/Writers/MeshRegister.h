
/*
 *  MeshRegister.cc:
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

static RegisterModule db1("Mesh", "MeshWriter", make_MeshWriter);
static RegisterModule db2("Writers", "MeshWriter", make_MeshWriter);

