
#ifndef UINTAH_HOMEBREW_ProcessorContext_H
#define UINTAH_HOMEBREW_ProcessorContext_H

namespace SCICore {
    namespace Thread {
	class SimpleReducer;
    }
}

class ProcessorContext {
public:
    static ProcessorContext* getRootContext();
    int numThreads() const {
	return d_numThreads;
    }
    int threadNumber() const {
	return d_threadNumber;
    }

    void setNumThreads(int numThreads) {
	d_numThreads = numThreads;
    }

    ProcessorContext* createContext(int threadNumber, int numThreads,
				    SCICore::Thread::SimpleReducer* reducer) const;
    ~ProcessorContext();

    void barrier_wait() const;
    double reduce_min(double) const;

private:
    const ProcessorContext* d_parent;
    SCICore::Thread::SimpleReducer* d_reducer;
    int d_numThreads;
    int d_threadNumber;

    ProcessorContext(const ProcessorContext* parent,
		     int threadNumber, int numThreads,
		     SCICore::Thread::SimpleReducer* reducer);

    ProcessorContext(const ProcessorContext&);
    ProcessorContext& operator=(const ProcessorContext&);
};

#endif
