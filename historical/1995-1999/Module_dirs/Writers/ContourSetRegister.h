
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

static RegisterModule db1("Contours", "ContourSetWriter", make_ContourSetWriter);
static RegisterModule db2("Writers", "ContourSetWriter", make_ContourSetWriter);
static RegisterModule db3("Dave", "ContourSetWriter", make_ContourSetWriter);

