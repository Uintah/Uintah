
#include "ThreadTopology.h"
#include <stdio.h>

/*
 * Simple interface for defining a complex topology for inter-thread
 * communication.  For a set of <tt>nthreads</tt> threads, the
 * topology is constructed by connecting one thread to another thread
 * with an optional <tt>weight</tt>.  Weights have arbitrary value,
 * and their relative magnitude expresses the importantance of keeping
 * two threads near each other when mapping the threads to physical
 * processors.
 */

ThreadTopology::ThreadTopology(int nthreads) {
    fprintf(stderr, "ThreadTopology not finished\n");
}

ThreadTopology::~ThreadTopology() {
    fprintf(stderr, "ThreadTopology not finished\n");
}

void ThreadTopology::connect(int from, int to, unsigned int weight) {
    fprintf(stderr, "ThreadTopology not finished\n");
}

