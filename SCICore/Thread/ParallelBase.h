
#ifndef SCI_THREAD_PARALLELBASE_H
#define SCI_THREAD_PARALLELBASE_H 1

/**************************************
 
CLASS
   ParallelBase
   
KEYWORDS
   ParallelBase
   
DESCRIPTION
   Helper class for Parallel class.  This will never be used
   by a user program.  See <b>Parallel</b> instead.
PATTERNS


WARNING
   
****************************************/

class ParallelBase {
protected:
    ParallelBase();
    virtual ~ParallelBase();
    friend class Thread;
public:
    //////////
    // <i>The thread body</i>
    virtual void run(int proc)=0;
};

#endif


