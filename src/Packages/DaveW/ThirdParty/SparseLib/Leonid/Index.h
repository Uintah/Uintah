/****************************************************************
 *  Class Index.h                                               *
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


#ifndef INDEX_H
#define INDEX_H 1

#include <iostream.h>

class Index
{
private:
  int start_;
  int end_;

public:
//constructors:
  Index():start_(0),end_(0){}
  Index(int start,int end): start_(start), end_(end) {}
  Index(const Index &I): start_(I.start_), end_(I.end_) {}

//operations:
  inline int& start(){return (start_);}
  inline int& end(){return (end_);}
  inline int length(){return (end_ - start_ + 1);}
  inline Index &operator=(const Index & I){
    start_ = I.start_; end_ = I.end_;return(*this);}

  inline friend ostream& operator<<(ostream& output, Index& I){
    output << "(" << I.start() << ":"  << I.end() << ")";
    return output;}

};
#endif
