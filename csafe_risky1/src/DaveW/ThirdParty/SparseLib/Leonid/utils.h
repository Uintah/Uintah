/****************************************************************
 *  utils.h                                                     *
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


#ifndef UTILS_H
#define UTILS_H 1

//-----------------------------------------------------------------
inline void check_rows(int& row_begin, int& row_end,int nrows){

 if( (row_end == 0) || (row_end > nrows)) row_end = nrows;
  if (row_begin < 0) row_begin = 0;


}
//-----------------------------------------------------------------
inline void check_cols(int& col_begin, int& col_end,int ncols){

 if( (col_end == 0) || (col_end > ncols)) col_end = ncols;
  if (col_begin < 0) col_begin = 0;


}


//-----------------------------------------------------------------
/*
inline int search(int i,int a[],int l,int r){
  int x;
  while(r>=l){
    x = (l+r)/2;
    if( i < a[x]) r = x-1;
    else l = x+1;
    if (i == a[x]) return x;
  }
  return (-1); 
}
*/
//-----------------------------------------------------------------
inline int search(int i,int a[],int l,int r){
  
int k;

  for(k = l;k<r;k++){
    if (a[k] == i)
      return k; 
  }
  
  return (-1); 
}
//-----------------------------------------------------------------

inline int ZMin(int d1, int d2)
{
    return d1<d2?d1:d2;
}

//-----------------------------------------------------------------
inline int ZMax(int d1, int d2)
{
    return d1>d2?d1:d2;
}

//-----------------------------------------------------------------


#endif

