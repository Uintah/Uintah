#ifndef Nebo_Semaphore_h
#define Nebo_Semaphore_h

#include <boost/thread/mutex.hpp>
#include <boost/thread/condition_variable.hpp>

namespace SpatialOps{

  /**
   * @struct Semaphore
   *
   * @brief Provide resource management for multithreaded situations.
   *
   * Implemented to replace boost::interprocess::interprocess_semaphore due
   * to posix semaphores bug in glibc: http://sourceware.org/bugzilla/show_bug.cgi?id=12674
   *
   * However, there are many more bugs in glibc! On some (non x86 / x86_64)
   * platforms, this one could bite us: http://sourceware.org/bugzilla/show_bug.cgi?id=13690
   */
  struct Semaphore {
  public:

    /**
     * @param initial the initial number of resources available
     */
    Semaphore(int initial=0) : val_(initial){}

    /** @brief release a resource */
    inline void post() {
      boost::lock_guard<boost::mutex> lock(mut_);
      ++val_;
      cond_.notify_one();
    }

    /**
     * @brief Wait until a resource is available (a call to \c post is made).
     */
    inline void wait() {
      boost::unique_lock<boost::mutex> lock(mut_);
      while (val_ <= 0) {
        cond_.wait(lock);
      }
      --val_;
    }

  private:
    boost::condition_variable cond_;
    boost::mutex mut_;
    int val_;
  };
}
#endif // Nebo_Semaphore_h
