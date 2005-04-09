
/*
 *  TrigTable.h: Faster ways to do trig...
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   May 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Math_TrigTable_h
#define SCI_Math_TrigTable_h 1

class SinCosTable {
    double* sindata;
    double* cosdata;
    int n;
public:
    SinCosTable(int n, double min, double max, double scale=1.0);
    ~SinCosTable();
    double sin(int) const;
    double cos(int) const;
};

#endif
