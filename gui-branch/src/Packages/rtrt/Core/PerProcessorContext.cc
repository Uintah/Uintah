

#include "PerProcessorContext.h"

using namespace rtrt;

PerProcessorContext::PerProcessorContext(int size, int scratchsize)
  : datasize(size), scratchsize(scratchsize)
{
  data=new char[size];
  scratch=new char[scratchsize];
}

PerProcessorContext::~PerProcessorContext()
{
    delete[] data;
}

