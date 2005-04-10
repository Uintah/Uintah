
#include "Profiler.h"
#include "Thread.h"



void Profiler::run()
{
    Thread::profile(d_in, d_out);
}

Profiler::Profiler()
{
}

Profiler::Profiler(FILE* in, FILE* out)
    : d_in(in), d_out(out)
{
}

Profiler::~Profiler()
{
    if(d_in)
        fclose(d_in);
    if(d_out)
        fclose(d_out);
}

