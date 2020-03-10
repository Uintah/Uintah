/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef CORE_GRID_TASK_H
#define CORE_GRID_TASK_H

#include <Core/Geometry/IntVector.h>
#include <Core/Grid/Ghost.h>
#include <Core/Grid/TaskStatus.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Parallel/ExecutionObject.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Parallel/SpaceTypes.h>
#include <Core/Parallel/UintahParams.h>
#include <CCA/Ports/DataWarehouseP.h>
#include <Core/Util/constHandle.h>
#include <Core/Util/DOUT.hpp>
#include <Core/Util/TupleHelpers.hpp>

#include <set>
#include <vector>
#include <string>
#include <iostream>

namespace Uintah {

class Level;
class DataWarehouse;
class OnDemandDataWarehouse;
class ProcessorGroup;
class Task;
class DetailedTask;
class TaskStatus;

/**************************************

 CLASS
   Task

 GENERAL INFORMATION
   Task.h

 Steven G. Parker
 Department of Computer Science
 University of Utah

 Center for the Simulation of Accidental Fires and Explosions (C-SAFE)

 KEYWORDS
   Task

 DESCRIPTION

 ****************************************/

class Task {
 
public: // class Task

protected: // class Task

  // base Action class
  class ActionBase {

    public:

      virtual ~ActionBase(){};

      virtual void doit( const PatchSubset    * patches
                       , const MaterialSubset * matls
                       ,       DataWarehouse  * fromDW
                       ,       DataWarehouse  * toDW
                       ,       UintahParams   & uintahParams
                       ) = 0;
  };

// Nvidia's nvcc compiler version 8 and 9 have a bug where it can't compile std::tuples with more than 2 template parameters
// The following uses std::tuple if nvcc version 8 or 9 is NOT used, otherwise, the variatic templates are manually unwrapped.
// Not that we can can't use the unwrapped templates exclusively, Alan H mentioned the justification for going to variatic
// templates was to avoid another bug elsewhere.  -- Brad P. Jan 10 2018
#if !(defined (__CUDACC_VER_MAJOR__) && __CUDACC_VER_MAJOR__ >=8)

private: // class Task

  // CPU Action constructor
  template<typename T, typename... Args>
  class Action : public ActionBase {

      T * ptr;
      void (T::*pmf)( const ProcessorGroup * pg
                    , const PatchSubset    * patches
                    , const MaterialSubset * m_matls
                    ,       DataWarehouse  * fromDW
                    ,       DataWarehouse  * toDW
                    ,       Args...          args
                    );
      std::tuple<Args...> m_args;


  public: // class Action

    Action( T * ptr
          , void (T::*pmf)( const ProcessorGroup * pg
                          , const PatchSubset    * patches
                          , const MaterialSubset * m_matls
                          ,       DataWarehouse  * fromDW
                          ,       DataWarehouse  * toDW
                          ,       Args...          args
                          )
          , Args... args
          )
      : ptr(ptr)
      , pmf(pmf)
      , m_args(std::forward<Args>(args)...)
    {}

    virtual ~Action() {}

    //////////
    //
    virtual void doit( const PatchSubset    * patches
                     , const MaterialSubset * matls
                     ,       DataWarehouse  * fromDW
                     ,       DataWarehouse  * toDW
                     ,       UintahParams   & uintahParams
                     )
    {
      doit_impl(uintahParams.getProcessorGroup(), patches, matls, fromDW, toDW, typename Tuple::gens<sizeof...(Args)>::type());
    }


  private: // class Action

    template<int... S>
    void doit_impl( const ProcessorGroup * pg
                  , const PatchSubset    * patches
                  , const MaterialSubset * matls
                  ,       DataWarehouse  * fromDW
                  ,       DataWarehouse  * toDW
                  ,       Tuple::seq<S...>
                  )
    {
      (ptr->*pmf)(pg, patches, matls, fromDW, toDW, std::get<S>(m_args)...);
    }

  };  // end CPU Action class

  // Kokkos enabled task portable Action constructor
  template<typename T, typename ExecSpace, typename MemSpace, typename... Args>
  class ActionPortable : public ActionBase {

    T * ptr;
    void (T::*pmf)( const PatchSubset                          * patches
                  , const MaterialSubset                       * m_matls
                  ,       OnDemandDataWarehouse                * fromDW
                  ,       OnDemandDataWarehouse                * toDW
                  ,       UintahParams                         & uintahParams
                  ,       ExecutionObject<ExecSpace, MemSpace> & execObj
                  ,       Args...                                args
                  );
    std::tuple<Args...> m_args;


  public: // class ActionPortable

    ActionPortable( T * ptr
                  , void (T::*pmf)( const PatchSubset                          * patches
                                  , const MaterialSubset                       * m_matls
                                  ,       OnDemandDataWarehouse                * fromDW
                                  ,       OnDemandDataWarehouse                * toDW
                                  ,       UintahParams                         & uintahParams
                                  ,       ExecutionObject<ExecSpace, MemSpace> & execObj
                                  ,       Args...                                args
                                  )
                  , Args... args
                  )
      : ptr(ptr)
      , pmf(pmf)
      , m_args(std::forward<Args>(args)...)
    {}

    virtual ~ActionPortable() {}

    void doit( const PatchSubset    * patches
             , const MaterialSubset * matls
             ,       DataWarehouse  * fromDW
             ,       DataWarehouse  * toDW
             ,       UintahParams   & uintahParams
             )
    {
      ExecutionObject<ExecSpace, MemSpace> execObj;

      execObj.setCudaThreadsPerBlock(Uintah::Parallel::getCudaThreadsPerBlock());
      execObj.setCudaBlocksPerLoop(Uintah::Parallel::getCudaBlocksPerLoop());

      const int numStreams = uintahParams.getNumStreams();

      for (int i = 0; i < numStreams; i++) {
        execObj.setStream(uintahParams.getStream(i), 0);
      }

      doit_impl(patches, matls, fromDW, toDW, uintahParams, execObj, typename Tuple::gens<sizeof...(Args)>::type());
    }

  private : // class ActionPortable

    template<int... S>
    void doit_impl( const PatchSubset                          * patches
                  , const MaterialSubset                       * matls
                  ,       DataWarehouse                        * fromDW
                  ,       DataWarehouse                        * toDW
                  ,       UintahParams                         & uintahParams
                  ,       ExecutionObject<ExecSpace, MemSpace> & execObj
                  ,       Tuple::seq<S...>
                  )
      {
        (ptr->*pmf)(patches, matls, reinterpret_cast<OnDemandDataWarehouse*>(fromDW), reinterpret_cast<OnDemandDataWarehouse*>(toDW), uintahParams, execObj, std::get<S>(m_args)...);
      }

  };  // end Kokkos enabled task Action constructor

#else  // If using nvcc compiler which can't use Tuple<>

private:

  // begin old CPU only Action constructors
  template<class T>
  class Action : public ActionBase {

    T * ptr;
    void (T::*pmf)( const ProcessorGroup * pg
                  , const PatchSubset    * patches
                  , const MaterialSubset * matls
                  ,       DataWarehouse  * fromDW
                  ,       DataWarehouse  * toDW
                  );


  public: // class Action

    Action( T * ptr
          , void (T::*pmf)( const ProcessorGroup * pg
                          , const PatchSubset    * patches
                          , const MaterialSubset * matls
                          ,       DataWarehouse  * fromDW
                          ,       DataWarehouse  * toDW
                          )
          )
      : ptr(ptr)
      , pmf(pmf)
    {}

    virtual ~Action() {}

    //////////
    //
    virtual void doit( const PatchSubset    * patches
                     , const MaterialSubset * matls
                     ,       DataWarehouse  * fromDW
                     ,       DataWarehouse  * toDW
                     ,       UintahParams   & uintahParams
                     )
    {
      (ptr->*pmf)(uintahParams.getProcessorGroup(), patches, matls, fromDW, toDW);
    }

  };  // end class Action

  template<class T, class Arg1>
  class Action1 : public ActionBase {

    T * ptr;
    void (T::*pmf)( const ProcessorGroup * pg
                  , const PatchSubset    * patches
                  , const MaterialSubset * matls
                  ,       DataWarehouse  * fromDW
                  ,       DataWarehouse  * toDW
                  ,       Arg1             arg1
                  );
    Arg1 arg1;


  public: // class Action1

    Action1( T * ptr
           , void (T::*pmf)( const ProcessorGroup * pg
                           , const PatchSubset    * patches
                           , const MaterialSubset * matls
                           ,       DataWarehouse  * fromDW
                           ,       DataWarehouse  * toDW
                           ,       Arg1             arg1
                           )
           , Arg1 arg1
           )
      : ptr(ptr)
      , pmf(pmf)
      , arg1(arg1)
    {}

    virtual ~Action1() {}

    //////////
    //
    virtual void doit( const PatchSubset    * patches
                     , const MaterialSubset * matls
                     ,       DataWarehouse  * fromDW
                     ,       DataWarehouse  * toDW
                     ,       UintahParams   & uintahParams
                     )
    {
      (ptr->*pmf)(uintahParams.getProcessorGroup(), patches, matls, fromDW, toDW, arg1);
    }

  };  // end class Action1

  template<class T, class Arg1, class Arg2>
  class Action2 : public ActionBase {

    T * ptr;
    void (T::*pmf)( const ProcessorGroup * pg
                  , const PatchSubset    * patches
                  , const MaterialSubset * matls
                  ,       DataWarehouse  * fromDW
                  ,       DataWarehouse  * toDW
                  ,       Arg1             arg1
                  ,       Arg2             arg2
                  );
    Arg1 arg1;
    Arg2 arg2;


  public: // class Action2

    Action2( T * ptr
           , void (T::*pmf)( const ProcessorGroup *
                           , const PatchSubset    * patches
                           , const MaterialSubset * matls
                           ,       DataWarehouse  *
                           ,       DataWarehouse  *
                           ,       Arg1
                           ,       Arg2
                           )
           , Arg1 arg1
           , Arg2 arg2
           )
      : ptr(ptr)
      , pmf(pmf)
      , arg1(arg1)
      , arg2(arg2)
    {}

    virtual ~Action2() {}

    //////////
    //
    virtual void doit( const PatchSubset    * patches
                     , const MaterialSubset * matls
                     ,       DataWarehouse  * fromDW
                     ,       DataWarehouse  * toDW
                     ,       UintahParams   & uintahParams
                     )
    {
      (ptr->*pmf)(uintahParams.getProcessorGroup(), patches, matls, fromDW, toDW, arg1, arg2);
    }

  };  // end class Action2

  template<class T, class Arg1, class Arg2, class Arg3>
  class Action3 : public ActionBase {

    T * ptr;
    void (T::*pmf)( const ProcessorGroup * pg
                  , const PatchSubset    * patches
                  , const MaterialSubset * matls
                  ,       DataWarehouse  * fromDW
                  ,       DataWarehouse  * toDW
                  ,       Arg1             arg1
                  ,       Arg2             arg2
                  ,       Arg3             arg3
                  );
    Arg1 arg1;
    Arg2 arg2;
    Arg3 arg3;


  public: // class Action3

    Action3( T * ptr
           , void (T::*pmf)( const ProcessorGroup * pg
                           , const PatchSubset    * patches
                           , const MaterialSubset * matls
                           ,       DataWarehouse  * fromDW
                           ,       DataWarehouse  * toDW
                           ,       Arg1
                           ,       Arg2
                           ,       Arg3
                           )
           , Arg1 arg1
           , Arg2 arg2
           , Arg3 arg3
           )
      : ptr(ptr)
      , pmf(pmf)
      , arg1(arg1)
      , arg2(arg2)
      , arg3(arg3)
    {}

    virtual ~Action3() {}

    //////////
    //
    virtual void doit( const PatchSubset    * patches
                     , const MaterialSubset * matls
                     ,       DataWarehouse  * fromDW
                     ,       DataWarehouse  * toDW
                     ,       UintahParams   & uintahParams
                     )
    {
      (ptr->*pmf)(uintahParams.getProcessorGroup(), patches, matls, fromDW, toDW, arg1, arg2, arg3);
    }

  };  // end Action3

  template<class T, class Arg1, class Arg2, class Arg3, class Arg4>
  class Action4 : public ActionBase {

    T * ptr;
    void (T::*pmf)( const ProcessorGroup * pg
                  , const PatchSubset    * patches
                  , const MaterialSubset * matls
                  ,       DataWarehouse  * fromDW
                  ,       DataWarehouse  * toDW
                  ,       Arg1             arg1
                  ,       Arg2             arg2
                  ,       Arg3             arg3
                  ,       Arg4             arg4
                  );
    Arg1 arg1;
    Arg2 arg2;
    Arg3 arg3;
    Arg4 arg4;


  public: // class Action4

    Action4( T * ptr
           , void (T::*pmf)( const ProcessorGroup * pg
                           , const PatchSubset    * patches
                           , const MaterialSubset * matls
                           ,       DataWarehouse  * fromDW
                           ,       DataWarehouse  * toDW
                           ,       Arg1
                           ,       Arg2
                           ,       Arg3
                           ,       Arg4
                           )
           , Arg1 arg1
           , Arg2 arg2
           , Arg3 arg3
           , Arg4 arg4
           )
      : ptr(ptr)
      , pmf(pmf)
      , arg1(arg1)
      , arg2(arg2)
      , arg3(arg3)
      , arg4(arg4)
    {}

    virtual ~Action4() {}

    //////////
    //
    virtual void doit( const PatchSubset    * patches
                     , const MaterialSubset * matls
                     ,       DataWarehouse  * fromDW
                     ,       DataWarehouse  * toDW
                     ,       UintahParams   & uintahParams
                     )
    {
      (ptr->*pmf)(uintahParams.getProcessorGroup(), patches, matls, fromDW, toDW, arg1, arg2, arg3, arg4);
    }

  };  // end Action4

  template<class T, class Arg1, class Arg2, class Arg3, class Arg4, class Arg5>
  class Action5 : public ActionBase {

    T * ptr;
    void (T::*pmf)( const ProcessorGroup * pg
                  , const PatchSubset    * patches
                  , const MaterialSubset * matls
                  ,       DataWarehouse  * fromDW
                  ,       DataWarehouse  * toDW
                  ,       Arg1             arg1
                  ,       Arg2             arg2
                  ,       Arg3             arg3
                  ,       Arg4             arg4
                  ,       Arg5             arg5
                  );
    Arg1 arg1;
    Arg2 arg2;
    Arg3 arg3;
    Arg4 arg4;
    Arg5 arg5;


  public: // class Action5

    Action5( T * ptr
           , void (T::*pmf)( const ProcessorGroup * pg
                           , const PatchSubset    * patches
                           , const MaterialSubset * matls
                           ,       DataWarehouse  * fromDW
                           ,       DataWarehouse  * toDW
                           ,       Arg1
                           ,       Arg2
                           ,       Arg3
                           ,       Arg4
                           ,       Arg5
                           )
           , Arg1 arg1
           , Arg2 arg2
           , Arg3 arg3
           , Arg4 arg4
           , Arg5 arg5
           )
      : ptr(ptr)
      , pmf(pmf)
      , arg1(arg1)
      , arg2(arg2)
      , arg3(arg3)
      , arg4(arg4)
      , arg5(arg5)
    {}

    virtual ~Action5() {}

    //////////
    //
    virtual void doit( const PatchSubset    * patches
                     , const MaterialSubset * matls
                     ,       DataWarehouse  * fromDW
                     ,       DataWarehouse  * toDW
                     ,       UintahParams   & uintahParams
                     )
    {
      (ptr->*pmf)(uintahParams.getProcessorGroup(), patches, matls, fromDW, toDW, arg1, arg2, arg3, arg4, arg5);
    }

  };  // end Action5

  // end old CPU only Action constructors

  // ------------------------------------------------------------------------

  // begin Portable Action constructors
  template<class T, typename ExecSpace, typename MemSpace>
  class ActionPortable : public ActionBase {

    T * ptr;
    void (T::*pmf)( const PatchSubset                          * patches
                  , const MaterialSubset                       * matls
                  ,       OnDemandDataWarehouse                * fromDW
                  ,       OnDemandDataWarehouse                * toDW
                  ,       UintahParams                         & uintahParams
                  ,       ExecutionObject<ExecSpace, MemSpace> & execObj
                  );


  public: // class ActionPortable

    ActionPortable( T * ptr
                  , void (T::*pmf)( const PatchSubset                          * patches
                                  , const MaterialSubset                       * matls
                                  ,       OnDemandDataWarehouse                * fromDW
                                  ,       OnDemandDataWarehouse                * toDW
                                  ,       UintahParams                         & uintahParams
                                  ,       ExecutionObject<ExecSpace, MemSpace> & execObj
                                  )
                  )
      : ptr(ptr)
      , pmf(pmf)
    {}

    virtual ~ActionPortable() {}

    //////////
    //
    virtual void doit( const PatchSubset    * patches
                     , const MaterialSubset * matls
                     ,       DataWarehouse  * fromDW
                     ,       DataWarehouse  * toDW
                     ,       UintahParams   & uintahParams
                     )
    {
      ExecutionObject<ExecSpace, MemSpace> execObj;

      execObj.setCudaThreadsPerBlock(Uintah::Parallel::getCudaThreadsPerBlock());
      execObj.setCudaBlocksPerLoop(Uintah::Parallel::getCudaBlocksPerLoop());

      const int numStreams = uintahParams.getNumStreams();

      for (int i = 0; i < numStreams; i++) {
        execObj.setStream(uintahParams.getStream(i), 0);
      }

      (ptr->*pmf)(patches, matls, reinterpret_cast<OnDemandDataWarehouse*>(fromDW), reinterpret_cast<OnDemandDataWarehouse*>(toDW), uintahParams, execObj);
    }

  };  // end class ActionPortable

  template<class T, typename ExecSpace, typename MemSpace, class Arg1>
  class ActionPortable1 : public ActionBase {

    T * ptr;
    void (T::*pmf)( const PatchSubset                          * patches
                  , const MaterialSubset                       * matls
                  ,       OnDemandDataWarehouse                * fromDW
                  ,       OnDemandDataWarehouse                * toDW
                  ,       UintahParams                         & uintahParams
                  ,       ExecutionObject<ExecSpace, MemSpace> & execObj
                  ,       Arg1                                   arg1
                  );
    Arg1 arg1;


  public: // class ActionPortable1

    ActionPortable1( T * ptr
                   , void (T::*pmf)( const PatchSubset                          * patches
                                   , const MaterialSubset                       * matls
                                   ,       OnDemandDataWarehouse                * fromDW
                                   ,       OnDemandDataWarehouse                * toDW
                                   ,       UintahParams                         & uintahParams
                                   ,       ExecutionObject<ExecSpace, MemSpace> & execObj
                                   ,       Arg1                                   arg1
                                   )
                   , Arg1 arg1
                   )
      : ptr(ptr)
      , pmf(pmf)
      , arg1(arg1)
    {}

    virtual ~ActionPortable1() {}

    //////////
    //
    virtual void doit( const PatchSubset    * patches
                     , const MaterialSubset * matls
                     ,       DataWarehouse  * fromDW
                     ,       DataWarehouse  * toDW
                     ,       UintahParams   & uintahParams
                     )
    {
      ExecutionObject<ExecSpace, MemSpace> execObj;

      execObj.setCudaThreadsPerBlock(Uintah::Parallel::getCudaThreadsPerBlock());
      execObj.setCudaBlocksPerLoop(Uintah::Parallel::getCudaBlocksPerLoop());

      const int numStreams = uintahParams.getNumStreams();

      for (int i = 0; i < numStreams; i++) {
        execObj.setStream(uintahParams.getStream(i), 0);
      }

      (ptr->*pmf)(patches, matls, reinterpret_cast<OnDemandDataWarehouse*>(fromDW), reinterpret_cast<OnDemandDataWarehouse*>(toDW), uintahParams, execObj, arg1);
    }

  };  // end class ActionPortable1

  template<class T, typename ExecSpace, typename MemSpace, class Arg1, class Arg2>
  class ActionPortable2 : public ActionBase {

    T * ptr;
    void (T::*pmf)( const PatchSubset                          * patches
                  , const MaterialSubset                       * matls
                  ,       OnDemandDataWarehouse                * fromDW
                  ,       OnDemandDataWarehouse                * toDW
                  ,       UintahParams                         & uintahParams
                  ,       ExecutionObject<ExecSpace, MemSpace> & execObj
                  ,       Arg1                                   arg1
                  ,       Arg2                                   arg2
                  );
    Arg1 arg1;
    Arg2 arg2;


  public: // class ActionPortable2

    ActionPortable2( T * ptr
                   , void (T::*pmf)( const PatchSubset                          * patches
                                   , const MaterialSubset                       * matls
                                   ,       OnDemandDataWarehouse                * fromDW
                                   ,       OnDemandDataWarehouse                * toDW
                                   ,       UintahParams                         & uintahParams
                                   ,       ExecutionObject<ExecSpace, MemSpace> & execObj
                                   ,       Arg1                                   arg1
                                   ,       Arg2                                   arg2
                                   )
                   , Arg1 arg1
                   , Arg2 arg2
                   )
      : ptr(ptr)
      , pmf(pmf)
      , arg1(arg1)
      , arg2(arg2)
    {}

    virtual ~ActionPortable2() {}

    //////////
    //
    virtual void doit( const PatchSubset    * patches
                     , const MaterialSubset * matls
                     ,       DataWarehouse  * fromDW
                     ,       DataWarehouse  * toDW
                     ,       UintahParams   & uintahParams
                     )
    {
      ExecutionObject<ExecSpace, MemSpace> execObj;

      execObj.setCudaThreadsPerBlock(Uintah::Parallel::getCudaThreadsPerBlock());
      execObj.setCudaBlocksPerLoop(Uintah::Parallel::getCudaBlocksPerLoop());

      const int numStreams = uintahParams.getNumStreams();

      for (int i = 0; i < numStreams; i++) {
        execObj.setStream(uintahParams.getStream(i), 0);
      }

      (ptr->*pmf)(patches, matls, reinterpret_cast<OnDemandDataWarehouse*>(fromDW), reinterpret_cast<OnDemandDataWarehouse*>(toDW), uintahParams, execObj, arg1, arg2);
    }

  };  // end class ActionPortable2

  template<class T, typename ExecSpace, typename MemSpace, class Arg1, class Arg2, class Arg3>
  class ActionPortable3 : public ActionBase {

    T * ptr;
    void (T::*pmf)( const PatchSubset                          * patches
                  , const MaterialSubset                       * matls
                  ,       OnDemandDataWarehouse                * fromDW
                  ,       OnDemandDataWarehouse                * toDW
                  ,       UintahParams                         & uintahParams
                  ,       ExecutionObject<ExecSpace, MemSpace> & execObj
                  ,       Arg1                                   arg1
                  ,       Arg2                                   arg2
                  ,       Arg3                                   arg3
                  );
    Arg1 arg1;
    Arg2 arg2;
    Arg3 arg3;


  public: // class ActionPortable3

    ActionPortable3( T * ptr
                   , void (T::*pmf)( const PatchSubset                          * patches
                                   , const MaterialSubset                       * matls
                                   ,       OnDemandDataWarehouse                * fromDW
                                   ,       OnDemandDataWarehouse                * toDW
                                   ,       UintahParams                         & uintahParams
                                   ,       ExecutionObject<ExecSpace, MemSpace> & execObj
                                   ,       Arg1                                   arg1
                                   ,       Arg2                                   arg2
                                   ,       Arg3                                   arg3
                                   )
                   , Arg1 arg1
                   , Arg2 arg2
                   , Arg3 arg3
                   )
      : ptr(ptr)
      , pmf(pmf)
      , arg1(arg1)
      , arg2(arg2)
      , arg3(arg3)
    {}

    virtual ~ActionPortable3() {}

    //////////
    //
    virtual void doit( const PatchSubset    * patches
                     , const MaterialSubset * matls
                     ,       DataWarehouse  * fromDW
                     ,       DataWarehouse  * toDW
                     ,       UintahParams   & uintahParams
                     )
    {
      ExecutionObject<ExecSpace, MemSpace> execObj;

      execObj.setCudaThreadsPerBlock(Uintah::Parallel::getCudaThreadsPerBlock());
      execObj.setCudaBlocksPerLoop(Uintah::Parallel::getCudaBlocksPerLoop());

      const int numStreams = uintahParams.getNumStreams();

      for (int i = 0; i < numStreams; i++) {
        execObj.setStream(uintahParams.getStream(i), 0);
      }

      (ptr->*pmf)(patches, matls, reinterpret_cast<OnDemandDataWarehouse*>(fromDW), reinterpret_cast<OnDemandDataWarehouse*>(toDW), uintahParams, execObj, arg1, arg2, arg3);
    }

  };  // end class ActionPortable3

  template<class T, typename ExecSpace, typename MemSpace, class Arg1, class Arg2, class Arg3, class Arg4>
  class ActionPortable4 : public ActionBase {

    T * ptr;
    void (T::*pmf)( const PatchSubset                          * patches
                  , const MaterialSubset                       * matls
                  ,       OnDemandDataWarehouse                * fromDW
                  ,       OnDemandDataWarehouse                * toDW
                  ,       UintahParams                         & uintahParams
                  ,       ExecutionObject<ExecSpace, MemSpace> & execObj
                  ,       Arg1                                   arg1
                  ,       Arg2                                   arg2
                  ,       Arg3                                   arg3
                  ,       Arg4                                   arg4
                  );
    Arg1 arg1;
    Arg2 arg2;
    Arg3 arg3;
    Arg4 arg4;


  public: // class ActionPortable4

    ActionPortable4( T * ptr
                   , void (T::*pmf)( const PatchSubset                          * patches
                                   , const MaterialSubset                       * matls
                                   ,       OnDemandDataWarehouse                * fromDW
                                   ,       OnDemandDataWarehouse                * toDW
                                   ,       UintahParams                         & uintahParams
                                   ,       ExecutionObject<ExecSpace, MemSpace> & execObj
                                   ,       Arg1                                   arg1
                                   ,       Arg2                                   arg2
                                   ,       Arg3                                   arg3
                                   ,       Arg4                                   arg4
                                   )
                   , Arg1 arg1
                   , Arg2 arg2
                   , Arg3 arg3
                   , Arg4 arg4
                   )
      : ptr(ptr)
      , pmf(pmf)
      , arg1(arg1)
      , arg2(arg2)
      , arg3(arg3)
      , arg4(arg4)
    {}

    virtual ~ActionPortable4() {}

    //////////
    //
    virtual void doit( const PatchSubset    * patches
                     , const MaterialSubset * matls
                     ,       DataWarehouse  * fromDW
                     ,       DataWarehouse  * toDW
                     ,       UintahParams   & uintahParams
                     )
    {
      ExecutionObject<ExecSpace, MemSpace> execObj;

      execObj.setCudaThreadsPerBlock(Uintah::Parallel::getCudaThreadsPerBlock());
      execObj.setCudaBlocksPerLoop(Uintah::Parallel::getCudaBlocksPerLoop());

      const int numStreams = uintahParams.getNumStreams();

      for (int i = 0; i < numStreams; i++) {
        execObj.setStream(uintahParams.getStream(i), 0);
      }

      (ptr->*pmf)(patches, matls, reinterpret_cast<OnDemandDataWarehouse*>(fromDW), reinterpret_cast<OnDemandDataWarehouse*>(toDW), uintahParams, execObj, arg1, arg2, arg3, arg4);
    }

  };  // end class ActionPortable4

  template<class T, typename ExecSpace, typename MemSpace, class Arg1, class Arg2, class Arg3, class Arg4, class Arg5>
  class ActionPortable5 : public ActionBase {

    T * ptr;
    void (T::*pmf)( const PatchSubset                          * patches
                  , const MaterialSubset                       * matls
                  ,       OnDemandDataWarehouse                * fromDW
                  ,       OnDemandDataWarehouse                * toDW
                  ,       UintahParams                         & uintahParams
                  ,       ExecutionObject<ExecSpace, MemSpace> & execObj
                  ,       Arg1                                   arg1
                  ,       Arg2                                   arg2
                  ,       Arg3                                   arg3
                  ,       Arg4                                   arg4
                  ,       Arg5                                   arg5
                  );
    Arg1 arg1;
    Arg2 arg2;
    Arg3 arg3;
    Arg4 arg4;
    Arg5 arg5;


  public:

    // class ActionPortable5
    ActionPortable5( T * ptr
                   , void (T::*pmf)( const PatchSubset                          * patches
                                   , const MaterialSubset                       * matls
                                   ,       OnDemandDataWarehouse                * fromDW
                                   ,       OnDemandDataWarehouse                * toDW
                                   ,       UintahParams                         & uintahParams
                                   ,       ExecutionObject<ExecSpace, MemSpace> & execObj
                                   ,       Arg1                                   arg1
                                   ,       Arg2                                   arg2
                                   ,       Arg3                                   arg3
                                   ,       Arg4                                   arg4
                                   ,       Arg5                                   arg5
                                   )
                   , Arg1 arg1
                   , Arg2 arg2
                   , Arg3 arg3
                   , Arg4 arg4
                   , Arg5 arg5
                   )
      : ptr(ptr)
      , pmf(pmf)
      , arg1(arg1)
      , arg2(arg2)
      , arg3(arg3)
      , arg4(arg4)
      , arg5(arg5)
    {}

    virtual ~ActionPortable5() {}

    //////////
    //
    virtual void doit( const PatchSubset    * patches
                     , const MaterialSubset * matls
                     ,       DataWarehouse  * fromDW
                     ,       DataWarehouse  * toDW
                     ,       UintahParams   & uintahParams
                     )
    {
      ExecutionObject<ExecSpace, MemSpace> execObj;

      execObj.setCudaThreadsPerBlock(Uintah::Parallel::getCudaThreadsPerBlock());
      execObj.setCudaBlocksPerLoop(Uintah::Parallel::getCudaBlocksPerLoop());

      const int numStreams = uintahParams.getNumStreams();

      for (int i = 0; i < numStreams; i++) {
        execObj.setStream(uintahParams.getStream(i), 0);
      }

      (ptr->*pmf)(patches, matls, reinterpret_cast<OnDemandDataWarehouse*>(fromDW), reinterpret_cast<OnDemandDataWarehouse*>(toDW), uintahParams, execObj, arg1, arg2, arg3, arg4, arg5);
    }

  };  // end class ActionPortable5

#endif // end #if !(defined (__CUDACC_VER_MAJOR__) && __CUDACC_VER_MAJOR__ >=8)


public: // class Task

  enum WhichDW {
     None        = -1
  ,  OldDW       = 0
  ,  NewDW       = 1
  ,  CoarseOldDW = 2
  ,  CoarseNewDW = 3
  ,  ParentOldDW = 4
  ,  ParentNewDW = 5
  ,  TotalDWs    = 6
  };

  enum {
      NoDW      = -1
    , InvalidDW = -2
  };

  enum TaskType {
      Normal
    , Reduction
    , InitialSend
    , OncePerProc // make sure to pass a PerProcessor PatchSet to the addTask function
    , Output
    , Spatial     // e.g. Radiometer task (spatial scheduling); must call task->setType(Task::Spatial)
    , Hypre       // previously identified as a OncePerProc
  };


  Task( const std::string & taskName, TaskType type )
    : m_task_name(taskName)
    , m_action(nullptr)
  {
    d_tasktype = type;
    initialize();
  }

// Nvidia's nvcc compiler version 8 and 9 have a bug where it can't compile std::tuples with more than 2 template parameters
// The following uses std::tuple if nvcc version 8 or 9 is NOT used, otherwise, the variatic templates are manually unwrapped.
// Not that we can can't use the unwrapped templates exclusively, Alan H mentioned the justification for going to variatic
// templates was to avoid another bug elsewhere.  -- Brad P. Jan 10 2018
#if !(defined (__CUDACC_VER_MAJOR__) && __CUDACC_VER_MAJOR__ >=8)

  // CPU Task constructor
  template<typename T, typename... Args>
  Task( const std::string & taskName
      , T * ptr
      , void (T::*pmf)( const ProcessorGroup *
                      , const PatchSubset    *
                      , const MaterialSubset *
                      ,       DataWarehouse  *
                      ,       DataWarehouse  *
                      ,       Args...
                      )
       , Args... args
      )
     : m_task_name(taskName)
     , m_action(scinew Action<T, Args...>(ptr, pmf, std::forward<Args>(args)...))
  {
    d_tasktype = Normal;
    initialize();
  }

  // Portable Task constructor
  template<typename T, typename ExecSpace, typename MemSpace, typename... Args>
  Task( const std::string & taskName
      , T * ptr
      , void (T::*pmf)( const PatchSubset    * patches
                      , const MaterialSubset * m_matls
                      ,       OnDemandDataWarehouse  * fromDW
                      ,       OnDemandDataWarehouse  * toDW
                      ,       UintahParams& uintahParams
                      ,       ExecutionObject<ExecSpace, MemSpace>& execObj
                      ,       Args...          args
                      )
      , Args... args
      )
      : m_task_name(taskName)
      , m_action(scinew ActionPortable<T, ExecSpace, MemSpace, Args...>(ptr, pmf, std::forward<Args>(args)...))
  {
    initialize();
    d_tasktype = Normal;
  }

#else

  // begin CPU only Task constructors
  template<class T>
  Task( const std::string & taskName
      , T * ptr
      , void (T::*pmf)( const ProcessorGroup * pg
                      , const PatchSubset    * patches
                      , const MaterialSubset * matls
                      ,       DataWarehouse  * fromDW
                      ,       DataWarehouse  * toDW
                      )
      )
    : m_task_name(taskName)
    , m_action(scinew Action<T>(ptr, pmf))
  {
    d_tasktype = Normal;
    initialize();
  }

  template<class T, class Arg1>
  Task( const std::string & taskName
      , T * ptr
      , void (T::*pmf)( const ProcessorGroup * pg
                      , const PatchSubset    * patches
                      , const MaterialSubset * matls
                      ,       DataWarehouse  * fromDW
                      ,       DataWarehouse  * toDW
                      ,       Arg1             arg1
                      )
      , Arg1 arg1
      )
    : m_task_name(taskName)
    , m_action(scinew Action1<T, Arg1>(ptr, pmf, arg1))
  {
    d_tasktype = Normal;
    initialize();
  }

  template<class T, class Arg1, class Arg2>
  Task( const std::string & taskName
      , T * ptr
      , void (T::*pmf)( const ProcessorGroup * pg
                      , const PatchSubset    * patches
                      , const MaterialSubset * matls
                      ,       DataWarehouse  * fromDW
                      ,       DataWarehouse  * toDW
                      ,       Arg1
                      ,       Arg2
                      )
      , Arg1 arg1
      , Arg2 arg2
      )
    : m_task_name(taskName)
    , m_action(scinew Action2<T, Arg1, Arg2>(ptr, pmf, arg1, arg2))
  {
    d_tasktype = Normal;
    initialize();
  }

  template<class T, class Arg1, class Arg2, class Arg3>
  Task( const std::string & taskName
      , T * ptr
      , void (T::*pmf)( const ProcessorGroup * pg
                      , const PatchSubset    * patches
                      , const MaterialSubset * matls
                      ,       DataWarehouse  * fromDW
                      ,       DataWarehouse  * toDW
                      ,       Arg1
                      ,       Arg2
                      ,       Arg3
                      )
      , Arg1 arg1
      , Arg2 arg2
      , Arg3 arg3
      )
    : m_task_name(taskName)
    , m_action(scinew Action3<T, Arg1, Arg2, Arg3>(ptr, pmf, arg1, arg2, arg3))
  {
    d_tasktype = Normal;
    initialize();
  }

  template<class T, class Arg1, class Arg2, class Arg3, class Arg4>
  Task( const std::string & taskName
      , T * ptr
      , void (T::*pmf)( const ProcessorGroup * pg
                      , const PatchSubset    * patches
                      , const MaterialSubset * matls
                      ,       DataWarehouse  * fromDW
                      ,       DataWarehouse  * toDW
                      ,       Arg1
                      ,       Arg2
                      ,       Arg3
                      ,       Arg4
                      )
      , Arg1 arg1
      , Arg2 arg2
      , Arg3 arg3
      , Arg4 arg4
      )
    : m_task_name(taskName)
    , m_action(scinew Action4<T, Arg1, Arg2, Arg3, Arg4>(ptr, pmf, arg1, arg2, arg3, arg4))
  {
    d_tasktype = Normal;
    initialize();
  }

  template<class T, class Arg1, class Arg2, class Arg3, class Arg4, class Arg5>
  Task( const std::string & taskName
      , T * ptr
      , void (T::*pmf)( const ProcessorGroup * pg
                      , const PatchSubset    * patches
                      , const MaterialSubset * matls
                      ,       DataWarehouse  * fromDW
                      ,       DataWarehouse  * toDW
                      ,       Arg1
                      ,       Arg2
                      ,       Arg3
                      ,       Arg4
                      ,       Arg5
                      )
      , Arg1 arg1
      , Arg2 arg2
      , Arg3 arg3
      , Arg4 arg4
      , Arg5 arg5
      )
    : m_task_name(taskName)
    , m_action(scinew Action5<T, Arg1, Arg2, Arg3, Arg4, Arg5>(ptr, pmf, arg1, arg2, arg3, arg4, arg5))
  {
    d_tasktype = Normal;
    initialize();
  }
  // end CPU only Task constructors

  // begin Portable Task constructors
  template<class T, typename ExecSpace, typename MemSpace>
  Task( const std::string & taskName
      , T * ptr
      , void (T::*pmf)( const PatchSubset                          * patches
                      , const MaterialSubset                       * matls
                      ,       OnDemandDataWarehouse                * fromDW
                      ,       OnDemandDataWarehouse                * toDW
                      ,       UintahParams                         & uintahParams
                      ,       ExecutionObject<ExecSpace, MemSpace> & execObj
                      )
      )
    : m_task_name(taskName)
    , m_action(scinew ActionPortable<T, ExecSpace, MemSpace>(ptr, pmf))
  {
    initialize();
    d_tasktype = Normal;
  }

  template<class T, typename ExecSpace, typename MemSpace, class Arg1>
  Task( const std::string & taskName
      , T * ptr
      , void (T::*pmf)( const PatchSubset                          * patches
                      , const MaterialSubset                       * matls
                      ,       OnDemandDataWarehouse                * fromDW
                      ,       OnDemandDataWarehouse                * toDW
                      ,       UintahParams                         & uintahParams
                      ,       ExecutionObject<ExecSpace, MemSpace> & execObj
                      ,       Arg1                                   arg1
                      )
      , Arg1 arg1
      )
    : m_task_name(taskName)
    , m_action(scinew ActionPortable1<T, ExecSpace, MemSpace, Arg1>(ptr, pmf, arg1))
  {
    initialize();
    d_tasktype = Normal;
  }

  template<class T, typename ExecSpace, typename MemSpace, class Arg1, class Arg2>
  Task( const std::string & taskName
      , T * ptr
      , void (T::*pmf)( const PatchSubset                          * patches
                      , const MaterialSubset                       * matls
                      ,       OnDemandDataWarehouse                * fromDW
                      ,       OnDemandDataWarehouse                * toDW
                      ,       UintahParams                         & uintahParams
                      ,       ExecutionObject<ExecSpace, MemSpace> & execObj
                      ,       Arg1                                   arg1
                      ,       Arg2                                   arg2
                      )
      , Arg1 arg1
      , Arg2 arg2
      )
    : m_task_name(taskName)
    , m_action(scinew ActionPortable2<T, ExecSpace, MemSpace, Arg1, Arg2>(ptr, pmf, arg1, arg2))
  {
    initialize();
    d_tasktype = Normal;
  }

  template<class T, typename ExecSpace, typename MemSpace, class Arg1, class Arg2, class Arg3>
  Task( const std::string & taskName
      , T * ptr
      , void (T::*pmf)( const PatchSubset                          * patches
                      , const MaterialSubset                       * matls
                      ,       OnDemandDataWarehouse                * fromDW
                      ,       OnDemandDataWarehouse                * toDW
                      ,       UintahParams                         & uintahParams
                      ,       ExecutionObject<ExecSpace, MemSpace> & execObj
                      ,       Arg1                                   arg1
                      ,       Arg2                                   arg2
                      ,       Arg3                                   arg3
                      )
      , Arg1 arg1
      , Arg2 arg2
      , Arg3 arg3
      )
    : m_task_name(taskName)
    , m_action(scinew ActionPortable3<T, ExecSpace, MemSpace, Arg1, Arg2, Arg3>(ptr, pmf, arg1, arg2, arg3))
  {
    initialize();
    d_tasktype = Normal;
  }

  template<class T, typename ExecSpace, typename MemSpace, class Arg1, class Arg2, class Arg3, class Arg4>
  Task( const std::string & taskName
      , T * ptr
      , void (T::*pmf)( const PatchSubset                          * patches
                      , const MaterialSubset                       * matls
                      ,       OnDemandDataWarehouse                * fromDW
                      ,       OnDemandDataWarehouse                * toDW
                      ,       UintahParams                         & uintahParams
                      ,       ExecutionObject<ExecSpace, MemSpace> & execObj
                      ,       Arg1                                   arg1
                      ,       Arg2                                   arg2
                      ,       Arg3                                   arg3
                      ,       Arg4                                   arg4
                      )
      , Arg1 arg1
      , Arg2 arg2
      , Arg3 arg3
      , Arg4 arg4
      )
    : m_task_name(taskName)
    , m_action(scinew ActionPortable4<T, ExecSpace, MemSpace, Arg1, Arg2, Arg3, Arg4>(ptr, pmf, arg1, arg2, arg3, arg4))
  {
    initialize();
    d_tasktype = Normal;
  }

  template<class T, typename ExecSpace, typename MemSpace, class Arg1, class Arg2, class Arg3, class Arg4, class Arg5>
  Task( const std::string & taskName
      , T * ptr
      , void (T::*pmf)( const PatchSubset                          * patches
                      , const MaterialSubset                       * matls
                      ,       OnDemandDataWarehouse                * fromDW
                      ,       OnDemandDataWarehouse                * toDW
                      ,       UintahParams                         & uintahParams
                      ,       ExecutionObject<ExecSpace, MemSpace> & execObj
                      ,       Arg1                                   arg1
                      ,       Arg2                                   arg2
                      ,       Arg3                                   arg3
                      ,       Arg4                                   arg4
                      ,       Arg5                                   arg5
                      )
      , Arg1 arg1
      , Arg2 arg2
      , Arg3 arg3
      , Arg4 arg4
      , Arg5 arg5
      )
    : m_task_name(taskName)
    , m_action(scinew ActionPortable5<T, ExecSpace, MemSpace, Arg1, Arg2, Arg3, Arg4, Arg5>(ptr, pmf, arg1, arg2, arg3, arg4, arg5))
  {
    initialize();
    d_tasktype = Normal;
  }

#endif  // end #if !(defined (__CUDACC_VER_MAJOR__) && __CUDACC_VER_MAJOR__ >=8)

  void initialize();

  virtual ~Task();

         void hasSubScheduler(bool state = true);
  inline bool getHasSubScheduler() const { return m_has_subscheduler; }

         void usesMPI(bool state);
  inline bool usesMPI() const { return m_uses_mpi; }

         void usesThreads(bool state);
  inline bool usesThreads() const { return m_uses_threads; }

         void setExecutionAndMemorySpace( const TaskAssignedExecutionSpace & executionSpaceTypeName
                                        , const TaskAssignedMemorySpace    & memorySpaceTypeName
                                        );

         TaskAssignedExecutionSpace getExecutionSpace() const;
         TaskAssignedMemorySpace    getMemorySpace() const;

         void usesDevice(bool state, int maxStreamsPerTask = -1);
  inline bool usesDevice() const { return m_uses_device; }
  inline int  maxStreamsPerTask() const { return  m_max_streams_per_task; }

         void usesSimVarPreloading(bool state);
  inline bool usesSimVarPreloading() const { return m_preload_sim_vars; }

         void usesKokkosOpenMP(bool state);
  inline bool usesKokkosOpenMP() const { return m_uses_kokkos_openmp; }

         void usesKokkosCuda(bool state);
  inline bool usesKokkosCuda() const { return m_uses_kokkos_cuda; }

  enum MaterialDomainSpec {
      NormalDomain  // <- Normal/default setting
    , OutOfDomain   // <- Require things from all material
  };

  enum PatchDomainSpec {
      ThisLevel        // <- Normal/default setting
    , CoarseLevel      // <- AMR :  The data on the coarse level under the range of the fine patches (including extra cells or boundary layers)
    , FineLevel        // <- AMR :  The data on the fine level over the range of the coarse patches (including extra cells or boundary layers)
    , OtherGridDomain  // for when we copy data to new grid after a regrid.
  };

  //////////
  // Most general case
  void requires( WhichDW
               , const VarLabel       *
               , const PatchSubset    * patches
               , PatchDomainSpec        patches_dom
               , int                    level_offset
               , const MaterialSubset * matls
               , MaterialDomainSpec     matls_dom
               , Ghost::GhostType       gtype
               , int                    numGhostCells = 0
               , bool                   oldTG         = false
               );

  //////////
  // Like general case, level_offset is not specified
  void requires(       WhichDW
               , const VarLabel           *
               , const PatchSubset        * patches
               ,       PatchDomainSpec      patches_dom
               , const MaterialSubset     * matls
               ,       MaterialDomainSpec   matls_dom
               ,       Ghost::GhostType     gtype
               ,       int                  numGhostCells = 0
               ,       bool                 oldTG         = false
               );

  //////////
  //
  void requires(       WhichDW
               , const VarLabel         *
               ,       Ghost::GhostType   gtype
               ,       int                numGhostCells = 0
               ,       bool               oldTG         = false
               );

  //////////
  //
  void requires(       WhichDW
               , const VarLabel         *
               , const PatchSubset      * patches
               , const MaterialSubset   * matls
               ,       Ghost::GhostType   gtype
               ,       int                numGhostCells = 0
               ,       bool               oldTG         = false
               );

  //////////
  //
  void requires( WhichDW
               , const VarLabel         *
               , const PatchSubset      * patches
               ,       Ghost::GhostType   gtype
               ,       int                numGhostCells = 0
               ,       bool               oldTG         = false
               );

  //////////
  //
  void requires(       WhichDW
               , const VarLabel        *
               , const MaterialSubset  * matls
               ,      Ghost::GhostType   gtype
               ,      int                numGhostCells = 0
               ,      bool               oldTG         = false);

  //////////
  //
  void requires(       WhichDW
               , const VarLabel           *
               , const MaterialSubset     * matls
               ,       MaterialDomainSpec   matls_dom
               ,       Ghost::GhostType     gtype
               ,       int                  numGhostCells = 0
               ,       bool                 oldTG         = false
               );

  //////////
  // Requires only for Reduction variables
  void requires(       WhichDW
               , const VarLabel           *
               , const Level              * level     = nullptr
               , const MaterialSubset     * matls     = nullptr
               ,       MaterialDomainSpec   matls_dom = NormalDomain
               ,       bool                 oldTG     = false
               );

  //////////
  // Requires for reduction variables or PerPatch variables
  void requires(       WhichDW
               , const VarLabel       *
               , const MaterialSubset * matls
               ,       bool             oldTG = false
               );

  //////////
  // Requires only for PerPatch variables
  void requires(       WhichDW
               , const VarLabel       *
               , const PatchSubset    * patches
               , const MaterialSubset * matls = nullptr
               );

  //////////
  // Most general case
  void computes( const VarLabel           *
               , const PatchSubset        * patches
               ,       PatchDomainSpec      patches_domain
               , const MaterialSubset     * matls
               ,       MaterialDomainSpec   matls_domain
               );

  //////////
  //
  void computes( const VarLabel       *
               , const PatchSubset    * patches = nullptr
               , const MaterialSubset * matls   = nullptr
               );

  //////////
  //
  void computes( const VarLabel       *
               , const MaterialSubset * matls
               );

  //////////
  //
  void computes( const VarLabel           *
               , const MaterialSubset     * matls
               ,       MaterialDomainSpec   matls_domain
               );

  //////////
  //
  void computes( const VarLabel        *
               , const PatchSubset     * patches
               ,       PatchDomainSpec   patches_domain
               );

  //////////
  //
  void computes( const VarLabel           *
               , const Level              * level
               , const MaterialSubset     * matls        = nullptr
               ,       MaterialDomainSpec   matls_domain = NormalDomain);

  //////////
  /*! \brief Allows a task to do a computes and modify with ghost cell specification.
   *
   *  \warning Uintah was built around the assumption that one is NOT allowed
      to compute or modify ghost cells. Therefore, it is unlawful in the Uintah sense
      to add a computes/modifies with ghost cells. However, certain components such
      as Wasatch break that design assumption from the point of view that,
      if a task can fill-in ghost values, then by all means do that and
      avoid an extra communication in the process. This, for example, is the
      case when one extrapolates data from the interior (e.g. Dynamic Smagorinsky
      model). Be aware that the ghost-values computed/modified in one patch will
      NOT be reproduced/correspond to interior cells of the neighboring patch,
      and vice versa.

      Another component which breaks this assumption is working with GPU tasks.
      Here it is not efficient to attempt to enlarge and copy variables within the
      GPU to make room for requires ghost cells.  Instead it is better to simply
      provide that extra room early when it's declared as a compute.  Then when it
      becomes a requires, no costly enlarging step is necessary.
   */
  void modifiesWithScratchGhost( const VarLabel           *
                               , const PatchSubset        * patches
                               ,       PatchDomainSpec      patches_domain
                               , const MaterialSubset     * matls
                               ,       MaterialDomainSpec   matls_domain
                               ,       Ghost::GhostType     gtype
                               ,       int                  numGhostCells
                               ,       bool                 oldTG = false
                               );

  void computesWithScratchGhost( const VarLabel           *
                               , const MaterialSubset     * matls
                               ,       MaterialDomainSpec   matls_domain
                               ,       Ghost::GhostType     gtype
                               ,       int                  numGhostCells
                               ,       bool                 oldTG = false
                               );

  //////////
  // Most general case
  void modifies( const VarLabel           *
               , const PatchSubset        * patches
               ,       PatchDomainSpec      patches_domain
               , const MaterialSubset     * matls
               ,       MaterialDomainSpec   matls_domain
               ,       bool                 oldTG = false
               );

  //////////
  //
  void modifies( const VarLabel       *
               , const PatchSubset    * patches
               , const MaterialSubset * matls
               ,       bool             oldTG = false
               );

  //////////
  //
  void modifies( const VarLabel       *
               , const MaterialSubset * matls
               ,       bool             oldTG = false
               );

  //////////
  //
  void modifies( const VarLabel       *
               , const MaterialSubset * matls
               , MaterialDomainSpec     matls_domain
               , bool                   oldTG = false
               );

  //////////
  //
  void modifies( const VarLabel *
               ,       bool     oldTG = false
               );

  //////////
  // Modify reduction vars
  void modifies( const VarLabel*
               , const Level* level
               , const MaterialSubset* matls = nullptr
               , MaterialDomainSpec matls_domain = NormalDomain
               , bool oldTG = false);

  //////////
  // Tells the task to actually execute the function assigned to it.
  virtual void doit( const PatchSubset                 *
                   , const MaterialSubset              *
                   ,       std::vector<DataWarehouseP> & dws
                   ,       UintahParams                & uintahParams
                   );

  inline const std::string & getName() const { return m_task_name; }

  inline const PatchSet * getPatchSet() const { return m_patch_set; }

  inline const MaterialSet * getMaterialSet() const { return m_matl_set; }

  bool hasDistalRequires() const;         // determines if this Task has any "distal" ghost cell requirements

  int m_phase{-1};                        // synchronized phase id, for dynamic task scheduling
  int m_comm{-1};                         // task communicator id, for threaded task scheduling
  std::map<int,int> m_max_ghost_cells;    // max ghost cells of this task
  int m_max_level_offset{0};              // max level offset of this task

  // Used in compiling the task graph with topological sort.
  //   Kept around for historical and reproducability reasons - APH, 04/05/19
  std::set<Task*> m_child_tasks;
  std::set<Task*> m_all_child_tasks;

  enum DepType {
      Modifies
    , Computes
    , Requires
  };

  struct Edge;

  struct Dependency {

      Dependency           * m_next{nullptr};
      DepType                m_dep_type;
      Task                 * m_task{nullptr};
      const VarLabel       * m_var{nullptr};
      bool                   m_look_in_old_tg;
      const PatchSubset    * m_patches{nullptr};
      const MaterialSubset * m_matls{nullptr};
      const Level          * m_reduction_level{nullptr};
      PatchDomainSpec        m_patches_dom{ThisLevel};
      MaterialDomainSpec     m_matls_dom;
      Ghost::GhostType       m_gtype{Ghost::None};
      WhichDW                m_whichdw;  // Used only by requires

      // in the multi-TG construct, this will signify that the required
      // m_var will be constructed by the old TG
      int m_num_ghost_cells{0};
      int m_level_offset{0};

      int mapDataWarehouse() const { return m_task->mapDataWarehouse(m_whichdw); }

      Dependency(       DepType              deptype
                ,       Task               * task
                ,       WhichDW              dw
                , const VarLabel           * var
                ,       bool                 oldtg
                , const PatchSubset        * patches
                , const MaterialSubset     * matls
                ,       PatchDomainSpec      patches_dom   = ThisLevel
                ,       MaterialDomainSpec   matls_dom     = NormalDomain
                ,       Ghost::GhostType     gtype         = Ghost::None
                ,       int                  numGhostCells = 0
                ,       int                  level_offset  = 0
                );

      Dependency(       DepType              deptype
                ,       Task               * task
                ,       WhichDW              dw
                , const VarLabel           * var
                ,       bool                 oldtg
                , const Level              * reductionLevel
                , const MaterialSubset     * matls
                ,       MaterialDomainSpec   matls_dom = NormalDomain
                );

      ~Dependency();

      constHandle<PatchSubset>
      getPatchesUnderDomain( const PatchSubset * domainPatches ) const;

      constHandle<MaterialSubset>
      getMaterialsUnderDomain( const MaterialSubset * domainMaterials ) const;

      // Used in compiling the task graph with topological sort.
      //   Kept around for historical and reproducability reasons - APH, 04/05/19
      Edge                 * m_req_head{nullptr};
      Edge                 * m_req_tail{nullptr};
      Edge                 * m_comp_head{nullptr};
      Edge                 * m_comp_tail{nullptr};
      inline void addComp( Edge * edge );
      inline void addReq(  Edge * edge );


  private:  // struct Dependency

      // eliminate copy, assignment and move
      Dependency( const Dependency & )            = delete;
      Dependency& operator=( const Dependency & ) = delete;
      Dependency( Dependency && )                 = delete;
      Dependency& operator=( Dependency && )      = delete;

      static constHandle<PatchSubset>

      getOtherLevelPatchSubset(       PatchDomainSpec   dom
                              ,       int               level_offset
                              , const PatchSubset     * subset
                              , const PatchSubset     * domainSubset
                              ,       int               ngc
                              );
  };  // end struct Dependency


  // Used in compiling the task graph with topological sort.
  //   Kept around for historical and reproducability reasons - APH, 04/05/19
  struct Edge {

     const Dependency * m_comp{nullptr};
     Edge             * m_comp_next{nullptr};
     const Dependency * m_req{nullptr};
     Edge             * m_req_next{nullptr};

     inline Edge( const Dependency * comp , const Dependency * req )
       : m_comp(comp)
       , m_req(req)
     {}

  }; // struct Edge



  const Dependency* getComputes() const { return m_comp_head; }

  const Dependency* getRequires() const { return m_req_head; }

  const Dependency* getModifies() const { return m_mod_head; }

  Dependency* getComputes() { return m_comp_head; }

  Dependency* getRequires() { return m_req_head; }

  Dependency* getModifies() { return m_mod_head; }

  // finds if it computes or modifies var
  bool hasComputes( const VarLabel * var
                  ,       int        matlIndex
                  , const Patch    * patch
                  ) const;

  // finds if it requires or modifies var
  bool hasRequires( const VarLabel          * var
                  ,       int                 matlIndex
                  , const Patch             * patch
                  ,       Uintah::IntVector   lowOffset
                  ,       Uintah::IntVector   highOffset
                  ,       WhichDW             dw
                  ) const;

  // finds if it modifies var
  bool hasModifies( const VarLabel * var
                  ,       int        matlIndex
                  , const Patch    * patch
                  ) const;

  bool isReductionTask() const { return d_tasktype == Reduction; }

  void setType(TaskType tasktype) { d_tasktype = tasktype; }

  TaskType getType() const { return d_tasktype; }

  //////////
  // Prints out information about the task...
  void display( std::ostream & out ) const;

  //////////
  // Prints out all information about the task, including dependencies
  void displayAll_DOUT( Uintah::Dout& dbg) const;
  
  void displayAll( std::ostream & out ) const;

  int mapDataWarehouse( WhichDW dw ) const;

  DataWarehouse* mapDataWarehouse( WhichDW dw, std::vector<DataWarehouseP> & dws ) const;

  int getSortedOrder() const { return m_sorted_order; }

  void setSortedOrder(int order) { m_sorted_order = order; }

  void setMapping(int dwmap[TotalDWs]);

  void setSets( const PatchSet * patches, const MaterialSet * matls );


private: // class Task

  using DepMap = std::multimap<const VarLabel*, Dependency*, VarLabel::Compare>;

  Dependency* isInDepMap( const DepMap   & depMap
                        , const VarLabel * var
                        ,       int        matlIndex
                        , const Patch    * patch
                        ) const;

  std::string m_task_name;

protected: // class Task

  ActionBase * m_action{nullptr};

  // eliminate copy, assignment and move
  Task( const Task & )            = delete;
  Task& operator=( const Task & ) = delete;
  Task( Task && )                 = delete;
  Task& operator=( Task && )      = delete;


  static const MaterialSubset* getGlobalMatlSubset();

  static MaterialSubset* globalMatlSubset;

  Dependency * m_comp_head{nullptr};
  Dependency * m_comp_tail{nullptr};
  Dependency * m_req_head{nullptr};
  Dependency * m_req_tail{nullptr};
  Dependency * m_mod_head{nullptr};
  Dependency * m_mod_tail{nullptr};

  DepMap       m_requires_old_dw;
  DepMap       m_computes;  // also contains modifies
  DepMap       m_requires;  // also contains modifies
  DepMap       m_modifies;

  const PatchSet    * m_patch_set{nullptr};
  const MaterialSet * m_matl_set{nullptr};

  bool m_uses_mpi{false};
  bool m_uses_threads{false};
  TaskAssignedExecutionSpace m_execution_space{};
  TaskAssignedMemorySpace    m_memory_space{};
  bool m_uses_device{false};
  bool m_preload_sim_vars{false};
  bool m_uses_kokkos_openmp{false};
  bool m_uses_kokkos_cuda{false};
  int  m_max_streams_per_task{1};
  bool m_subpatch_capable{false};
  bool m_has_subscheduler{false};

  TaskType d_tasktype;

  int m_dwmap[TotalDWs];
  int m_sorted_order{-1};

  friend std::ostream & operator <<(std::ostream & out, const Uintah::Task & task);
  friend std::ostream & operator <<(std::ostream & out, const Uintah::Task::TaskType & tt);
  friend std::ostream & operator <<(std::ostream & out, const Uintah::Task::Dependency & dep);

}; // end class Task


// Used in compiling the task graph with topological sort.
//   Kept around for historical and reproducability reasons - APH, 04/05/19
inline void Task::Dependency::addComp( Edge * edge )
{
  if (m_comp_tail) {
    m_comp_tail->m_comp_next = edge;
  }
  else {
    m_comp_head = edge;
  }
  m_comp_tail = edge;
}

inline void Task::Dependency::addReq( Edge * edge )
{
  if (m_req_tail) {
    m_req_tail->m_req_next = edge;
  }
  else {
    m_req_head = edge;
  }
  m_req_tail = edge;
}

}  // End namespace Uintah

// This must be at the bottom
#include <CCA/Ports/DataWarehouse.h>

#endif // CORE_GRID_TASK_H
