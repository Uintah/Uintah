/****************************************************************
 *  Class Matrix.h                                              *
 *                                                              *
 *  Written by:                                                 *
 *   Leonid Zhukov                                              *
 *   Department of Computer Science                             *
 *   University of Utah                                         *
 *   October 1997                                               *
 *                                                              *
 *  Copyright (C) 1997 SCI Group                                *
 *                                                              *
 *                                                              *
 ****************************************************************/

#ifndef ZMATRIX_H
#define ZMATRIX_H 1



template <class Data_Type>
class ZMatrix{
  
protected:
  int ncols;
  int nrows;

public:
//constructor:
  ZMatrix(): nrows(0),ncols(0){}

//operations:
  int nr() const {return nrows;}
  int nc() const {return ncols;}

  
  
};

#endif
