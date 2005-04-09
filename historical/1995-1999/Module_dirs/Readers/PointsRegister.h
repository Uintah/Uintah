
/*
 *  PointsRegister.h:
 *
 *  Written by:
 *   Carole Gitlin
 *   Department of Computer Science
 *   University of Utah
 *   April 1995
 *
 *  Copyright (C) 1995 SCI Group
 */

static RegisterModule db1("Mesh", "PointsReader", make_PointsReader);
static RegisterModule db2("Readers", "PointsReader", make_PointsReader);

