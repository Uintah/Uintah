
#include "Profiler.h"
#include "Thread.h"



void Profiler::run() {
    Thread::profile(in, out);
}

Profiler::Profiler() {
}

Profiler::Profiler(FILE* in, FILE* out) : in(in), out(out) {
}

Profiler::~Profiler() {
    if(in)
        fclose(in);
    if(out)
        fclose(out);
}

