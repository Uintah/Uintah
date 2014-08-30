#include <spatialops/ThreadPool.h>

namespace SpatialOps {

   //===========================================================================

   class ThreadPoolResourceManager {
     ThreadPoolResourceManager(){};
     ~ThreadPoolResourceManager(){};

   public:
     static ThreadPoolResourceManager& self();

     template<class VoidType> static bool insert( VoidType& rID, int threads );
     template<class VoidType> static bool remove( VoidType& rID, const int threads );
     template<class VoidType> static int resize( VoidType& rID, int threads );
     template<class VoidType> static int resize_active( VoidType& rID, const int threads );
     template<class VoidType> static int get_worker_count( VoidType& rID );
     template<class VoidType> static int get_max_active_worker_count( VoidType& rID );

   private:

     typedef std::map<void*, int> ResourceMap;
     typedef std::map<void*,int>::iterator ResourceIter;
     ResourceMap resourceMap_;

     class ExecutionMutex{
       const boost::mutex::scoped_lock lock;
       inline boost::mutex& get_mutex() const { static boost::mutex m; return m; }
     public:
       ExecutionMutex() : lock( get_mutex() ){}
       ~ExecutionMutex(){}
     };
   };

   ThreadPoolResourceManager&  ThreadPoolResourceManager::self()
   {
     static ThreadPoolResourceManager tprm;
     return tprm;
   }

   template<class VoidType>
   bool ThreadPoolResourceManager::insert( VoidType& rID, int threads )
   {
     ExecutionMutex lock;
     ThreadPoolResourceManager& tprm = ThreadPoolResourceManager::self();
     if( threads < 1 ) { threads = 1; }
     //Make sure we don't have the threadpool
     ResourceIter rit = tprm.resourceMap_.find(&rID);
     if ( rit == tprm.resourceMap_.end() ){
       tprm.resourceMap_.insert(std::make_pair(&rID, threads));
     } else {
       printf("Warning: attempting to insert a ThreadPool that already exists!\n");
       return false;
     }
     return true;
   }

   template<class VoidType>
   bool ThreadPoolResourceManager::remove( VoidType& rID, const int threads )
   {
     ExecutionMutex lock;
     ThreadPoolResourceManager& tprm = ThreadPoolResourceManager::self();
     ResourceIter rit = tprm.resourceMap_.find(&rID);
     if ( rit != tprm.resourceMap_.end() ){
       tprm.resourceMap_.erase(rit);
     } else {
       printf("Warning: attempting to remove ThreadPool that does not exist!\n");
       return false;
     }
     return true;
   }

   template<class VoidType>
   int ThreadPoolResourceManager::resize( VoidType& rID, int threads )
   {
     ExecutionMutex lock;
     ThreadPoolResourceManager& tprm = ThreadPoolResourceManager::self();
     //Make sure we have the threadpool
     ResourceIter rit = tprm.resourceMap_.find(&rID);
     if( rit == tprm.resourceMap_.end() ) {
       fprintf(stderr, "Error: ThreadPool does not exist!\n");
       return -1;
     }

     //Fast exit
     if( rit->second == threads ) { return threads; }

     //Connect the right resource interface
     VoidType* resource = (VoidType*)rit->first;

     if( threads < 1 ) { threads = 1; }
     rit->second = threads;
     resource->size_controller().resize(threads);
     return threads;
   }

   template<class VoidType>
   int ThreadPoolResourceManager::resize_active( VoidType& rID, const int threads )
   {
     ExecutionMutex lock;
     ThreadPoolResourceManager& tprm = ThreadPoolResourceManager::self();
     //Make sure we have the threadpool
     ResourceIter rit = tprm.resourceMap_.find(&rID);
     if( rit == tprm.resourceMap_.end() ) {
       fprintf(stderr, "Error: ThreadPool does not exist!\n");
       return -1;
     }
     //Connect the right resource interface
     VoidType* resource = (VoidType*)rit->first;
     resource->size_controller().resize_active(std::max(1,threads));
     return threads;
   }

   template<class VoidType>
   int ThreadPoolResourceManager::get_worker_count( VoidType& rID )
   {
     ExecutionMutex lock;
     ThreadPoolResourceManager& tprm = ThreadPoolResourceManager::self();
     ResourceIter rit = tprm.resourceMap_.find(&rID);
     if( rit == tprm.resourceMap_.end() ) {
       fprintf(stderr, "Error: Threadpool does not exist!\n");
       return -1;
     }
     VoidType* resource = (VoidType*)rit->first;
     return resource->size();
   }

   template<class VoidType>
   int ThreadPoolResourceManager::get_max_active_worker_count( VoidType& rID )
   {
     ExecutionMutex lock;
     ThreadPoolResourceManager& tprm = ThreadPoolResourceManager::self();
     //Make sure we have the threadpool
     ResourceIter rit = tprm.resourceMap_.find(&rID);
     if( rit == tprm.resourceMap_.end() ) {
       fprintf(stderr, "Error: ThreadPool does not exist!\n");
       return -1;
     }

     //Connect the right resource interface
     VoidType* resource = (VoidType*)rit->first;
     return resource->max_active();
   }

   //===========================================================================

   ThreadPool::ThreadPool( const int nthreads )
     : boost::threadpool::prio_pool( nthreads )
   {
     init_ = false;
   }

   ThreadPool::~ThreadPool()
   {}

   int ThreadPool::resize_pool( const int threadCount ){
     return ThreadPoolResourceManager::resize_active(self(),threadCount);
   }
   int ThreadPool::get_pool_size(){
     return ThreadPoolResourceManager::get_max_active_worker_count(self());
   }

   int ThreadPool::set_pool_capacity( const int threadCount ){
     return ThreadPoolResourceManager::resize(self(),threadCount);
   }
   int ThreadPool::get_pool_capacity(){
     return ThreadPoolResourceManager::get_worker_count(self());
   }

   ThreadPool& ThreadPool::self()
   {
     static ThreadPool tp(NTHREADS);
     ThreadPoolResourceManager& tprm = ThreadPoolResourceManager::self();
     if( tp.init_ == false ){
       tprm.insert<boost::threadpool::prio_pool>(tp, NTHREADS);
       tp.init_ = true;
     }
     return tp;
   }

   //===========================================================================

   ThreadPoolFIFO::ThreadPoolFIFO( const int nthreads )
     : boost::threadpool::fifo_pool( nthreads )
   {
     init_ = false;
   }

   ThreadPoolFIFO::~ThreadPoolFIFO()
   {}

   ThreadPoolFIFO&
   ThreadPoolFIFO::self()
   {
     static ThreadPoolFIFO tp( NTHREADS );
     ThreadPoolResourceManager& tprm = ThreadPoolResourceManager::self();
     if( tp.init_ == false ){
       tprm.insert<boost::threadpool::fifo_pool>( tp, NTHREADS );
       tp.init_ = true;
     }
     return tp;
   }

   int ThreadPoolFIFO::resize_pool( const int threadCount ){
     return ThreadPoolResourceManager::resize_active(self(),threadCount);
   }
   int ThreadPoolFIFO::get_pool_size(){
     return ThreadPoolResourceManager::get_max_active_worker_count(self());
   }

   int ThreadPoolFIFO::set_pool_capacity( const int threadCount ){
     return ThreadPoolResourceManager::resize(self(),threadCount);
   }
   int ThreadPoolFIFO::get_pool_capacity(){
     return ThreadPoolResourceManager::get_worker_count(self());
   }

   //===========================================================================

} // namespace SpatialOps
