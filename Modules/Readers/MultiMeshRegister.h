
/*
 *  MultiMeshRegister.cc:
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

static RegisterModule db1("MultiMesh", "MultiMeshReader", make_MultiMeshReader);
static RegisterModule db2("Readers", "MultiMeshReader", make_MultiMeshReader);

