

#include <Packages/rtrt/Core/PerProcessorContext.h>

using namespace rtrt;

PerProcessorContext::PerProcessorContext(int size, int scratchsize)
  : datasize(size), scratchsize(scratchsize)
{
  data=new char[size];
  for(int i=0;i<size;i++)
    data[i]=0;
  scratch=new char[scratchsize];
  for(int i=0;i<scratchsize;i++)
    scratch[i]=0;
}

PerProcessorContext::~PerProcessorContext()
{
    delete[] data;
}

