
/*
 *  ContourSetRegister.cc:
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   Jul 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

static RegisterModule db1("Contours", "ContourSetReader", make_ContourSetReader);
static RegisterModule db2("Readers", "ContourSetReader", make_ContourSetReader);

