/*
 * The MIT License
 *
 * Copyright (c) 1997-2016 The University of Utah
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

#ifndef UINTAH_HOMEBREW_Task_H
#define UINTAH_HOMEBREW_Task_H

#include <Core/Grid/Ghost.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <CCA/Ports/DataWarehouseP.h>
#include <Core/Util/constHandle.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Parallel/Parallel.h>

#include <map>
#include <set>
#include <vector>
#include <string>
#include <iostream>

namespace Uintah {

class Level;
class DataWarehouse;
class ProcessorGroup;

/**************************************

 CLASS
 Task

 Short description...

 GENERAL INFORMATION

 Task.h

 Steven G. Parker
 Department of Computer Science
 University of Utah

 Center for the Simulation of Accidental Fires and Explosions (C-SAFE)


 KEYWORDS
 Task

 DESCRIPTION
 Long description...

 WARNING

 ****************************************/

class Task {
 
  public: 
  enum CallBackEvent {
      CPU,    // <- normal CPU task, happens when a GPU enabled task runs on CPU
      preGPU, // <- pre GPU kernel callback, happens before CPU->GPU copy (reserved, not implemented yet... )
      GPU,    // <- GPU kernel callback, happens after dw: CPU->GPU copy, kernel launch should be queued in this callback
      postGPU // <- post GPU kernel callback, happens after dw: GPU->CPU copy but before MPI sends.
    };
 
  protected:

    // base Action class
    class ActionBase {
      public:
        virtual ~ActionBase();
        virtual void doit(CallBackEvent event,
                          const ProcessorGroup* pg,
                          const PatchSubset* patches,
                          const MaterialSubset* matls,
                          DataWarehouse* fromDW,
                          DataWarehouse* toDW,
                          void* oldTaskGpuDW,
                          void* newTaskGpuDW,
                          void* stream,
                          int deviceID) = 0;
    };

  private:

    // begin old CPU only Action constructors
    template<class T>
    class Action : public ActionBase {

        T* ptr;
        void (T::*pmf)(const ProcessorGroup* pg,
                       const PatchSubset* patches,
                       const MaterialSubset* matls,
                       DataWarehouse* fromDW,
                       DataWarehouse* toDW);
      public:
        // class Action
        Action(T* ptr,
               void (T::*pmf)(const ProcessorGroup* pg,
                              const PatchSubset* patches,
                              const MaterialSubset* matls,
                              DataWarehouse* fromDW,
                              DataWarehouse*toDW))
            : ptr(ptr), pmf(pmf)
        {
        }
        virtual ~Action()
        {
        }

        //////////
        // Insert Documentation Here:
        virtual void doit(CallBackEvent event,
                          const ProcessorGroup* pg,
                          const PatchSubset* patches,
                          const MaterialSubset* matls,
                          DataWarehouse* fromDW,
                          DataWarehouse* toDW,
                          void* oldTaskGpuDW,
                          void* newTaskGpuDW,
                          void* stream,
                          int deviceID)
        {
          (ptr->*pmf)(pg, patches, matls, fromDW, toDW);
        }
    };  // end class Action

    template<class T, class Arg1>
    class Action1 : public ActionBase {

        T* ptr;
        void (T::*pmf)(const ProcessorGroup* pg,
                       const PatchSubset* patches,
                       const MaterialSubset* matls,
                       DataWarehouse* fromDW,
                       DataWarehouse* toDW,
                       Arg1 arg1);
        Arg1 arg1;
      public:
        // class Action1
        Action1(T* ptr,
                void (T::*pmf)(const ProcessorGroup* pg,
                               const PatchSubset* patches,
                               const MaterialSubset* matls,
                               DataWarehouse* fromDW,
                               DataWarehouse* toDW,
                               Arg1 arg1),
                Arg1 arg1)
            : ptr(ptr), pmf(pmf), arg1(arg1)
        {
        }
        virtual ~Action1()
        {
        }

        //////////
        // Insert Documentation Here:
        virtual void doit(CallBackEvent event,
                          const ProcessorGroup* pg,
                          const PatchSubset* patches,
                          const MaterialSubset* matls,
                          DataWarehouse* fromDW,
                          DataWarehouse* toDW,
                          void* oldTaskGpuDW,
                          void* newTaskGpuDW,
                          void* stream,
                          int deviceID)
        {
          (ptr->*pmf)(pg, patches, matls, fromDW, toDW, arg1);
        }
    };  // end class Action1

    template<class T, class Arg1, class Arg2>
    class Action2 : public ActionBase {

        T* ptr;
        void (T::*pmf)(const ProcessorGroup* pg,
                       const PatchSubset* patches,
                       const MaterialSubset* matls,
                       DataWarehouse* fromDW,
                       DataWarehouse* toDW,
                       Arg1 arg1,
                       Arg2 arg2);
        Arg1 arg1;
        Arg2 arg2;
      public:
        // class Action2
        Action2(T* ptr,
                void (T::*pmf)(const ProcessorGroup*,
                               const PatchSubset* patches,
                               const MaterialSubset* matls,
                               DataWarehouse*,
                               DataWarehouse*,
                               Arg1,
                               Arg2),
                Arg1 arg1,
                Arg2 arg2)
            : ptr(ptr), pmf(pmf), arg1(arg1), arg2(arg2)
        {
        }
        virtual ~Action2()
        {
        }

        //////////
        // Insert Documentation Here:
        virtual void doit(CallBackEvent event,
                          const ProcessorGroup* pg,
                          const PatchSubset* patches,
                          const MaterialSubset* matls,
                          DataWarehouse* fromDW,
                          DataWarehouse* toDW,
                          void* oldTaskGpuDW,
                          void* newTaskGpuDW,
                          void* stream,
                          int deviceID)
        {
          (ptr->*pmf)(pg, patches, matls, fromDW, toDW, arg1, arg2);
        }
    };  // end class Action2

    template<class T, class Arg1, class Arg2, class Arg3>
    class Action3 : public ActionBase {

        T* ptr;
        void (T::*pmf)(const ProcessorGroup* pg,
                       const PatchSubset* patches,
                       const MaterialSubset* matls,
                       DataWarehouse* fromDW,
                       DataWarehouse* toDW,
                       Arg1 arg1,
                       Arg2 arg2,
                       Arg3 arg3);
        Arg1 arg1;
        Arg2 arg2;
        Arg3 arg3;
      public:
        // class Action3
        Action3(T* ptr,
                void (T::*pmf)(const ProcessorGroup* pg,
                               const PatchSubset* patches,
                               const MaterialSubset* matls,
                               DataWarehouse* fromDW,
                               DataWarehouse* toDW,
                               Arg1,
                               Arg2,
                               Arg3),
                Arg1 arg1,
                Arg2 arg2,
                Arg3 arg3)
            : ptr(ptr), pmf(pmf), arg1(arg1), arg2(arg2), arg3(arg3)
        {
        }
        virtual ~Action3()
        {
        }

        //////////
        // Insert Documentation Here:
        virtual void doit(CallBackEvent event,
                          const ProcessorGroup* pg,
                          const PatchSubset* patches,
                          const MaterialSubset* matls,
                          DataWarehouse* fromDW,
                          DataWarehouse* toDW,
                          void* oldTaskGpuDW,
                          void* newTaskGpuDW,
                          void* stream,
                          int deviceID)
        {
          (ptr->*pmf)(pg, patches, matls, fromDW, toDW, arg1, arg2, arg3);
        }
    };  // end Action3

    template<class T, class Arg1, class Arg2, class Arg3, class Arg4>
    class Action4 : public ActionBase {

        T* ptr;
        void (T::*pmf)(const ProcessorGroup* pg,
                       const PatchSubset* patches,
                       const MaterialSubset* matls,
                       DataWarehouse* fromDW,
                       DataWarehouse* toDW,
                       Arg1 arg1,
                       Arg2 arg2,
                       Arg3 arg3,
                       Arg4 arg4);
        Arg1 arg1;
        Arg2 arg2;
        Arg3 arg3;
        Arg4 arg4;
      public:
        // class Action4
        Action4(T* ptr,
                void (T::*pmf)(const ProcessorGroup* pg,
                               const PatchSubset* patches,
                               const MaterialSubset* matls,
                               DataWarehouse* fromDW,
                               DataWarehouse* toDW,
                               Arg1,
                               Arg2,
                               Arg3,
                               Arg4),
                Arg1 arg1,
                Arg2 arg2,
                Arg3 arg3,
                Arg4 arg4)
            : ptr(ptr), pmf(pmf), arg1(arg1), arg2(arg2), arg3(arg3), arg4(arg4)
        {
        }
        virtual ~Action4()
        {
        }

        //////////
        // Insert Documentation Here:
        virtual void doit(CallBackEvent event,
                          const ProcessorGroup* pg,
                          const PatchSubset* patches,
                          const MaterialSubset* matls,
                          DataWarehouse* fromDW,
                          DataWarehouse* toDW,
                          void* oldTaskGpuDW,
                          void* newTaskGpuDW,
                          void* stream,
                          int deviceID)
        {
          (ptr->*pmf)(pg, patches, matls, fromDW, toDW, arg1, arg2, arg3, arg4);
        }
    };  // end Action4

    template<class T, class Arg1, class Arg2, class Arg3, class Arg4, class Arg5>
    class Action5 : public ActionBase {

        T* ptr;
        void (T::*pmf)(const ProcessorGroup* pg,
                       const PatchSubset* patches,
                       const MaterialSubset* matls,
                       DataWarehouse* fromDW,
                       DataWarehouse* toDW,
                       Arg1 arg1,
                       Arg2 arg2,
                       Arg3 arg3,
                       Arg4 arg4,
                       Arg5 arg5);
        Arg1 arg1;
        Arg2 arg2;
        Arg3 arg3;
        Arg4 arg4;
        Arg5 arg5;
      public:
        // class Action5
        Action5(T* ptr,
                void (T::*pmf)(const ProcessorGroup* pg,
                               const PatchSubset* patches,
                               const MaterialSubset* matls,
                               DataWarehouse* fromDW,
                               DataWarehouse* toDW,
                               Arg1,
                               Arg2,
                               Arg3,
                               Arg4,
                               Arg5),
                Arg1 arg1,
                Arg2 arg2,
                Arg3 arg3,
                Arg4 arg4,
                Arg5 arg5)
            : ptr(ptr), pmf(pmf), arg1(arg1), arg2(arg2), arg3(arg3), arg4(arg4), arg5(arg5)
        {
        }
        virtual ~Action5()
        {
        }

        //////////
        // Insert Documentation Here:
        virtual void doit(CallBackEvent event,
                          const ProcessorGroup* pg,
                          const PatchSubset* patches,
                          const MaterialSubset* matls,
                          DataWarehouse* fromDW,
                          DataWarehouse* toDW,
                          void* oldTaskGpuDW,
                          void* newTaskGpuDW,
                          void* stream,
                          int deviceID)
        {
          (ptr->*pmf)(pg, patches, matls, fromDW, toDW, arg1, arg2, arg3, arg4, arg5);
        }
    };  // end Action5
    // end old CPU only Action constructors

    // ------------------------------------------------------------------------

    // begin Device Action constructors
    template<class T>
    class ActionDevice : public ActionBase {
        T* ptr;
        void (T::*pmf)(CallBackEvent event,
                       const ProcessorGroup* pg,
                       const PatchSubset* patches,
                       const MaterialSubset* matls,
                       DataWarehouse* fromDW,
                       DataWarehouse* toDW,
                       void* oldTaskGpuDW,
                       void* newTaskGpuDW,
                       void* stream,
                       int deviceID);
      public:
        // class ActionDevice
        ActionDevice( T * ptr,
                      void (T::*pmf)(CallBackEvent event,
                                     const ProcessorGroup* pg,
                                     const PatchSubset* patches,
                                     const MaterialSubset* matls,
                                     DataWarehouse* fromDW,
                                     DataWarehouse* toDW,
                                     void* oldTaskGpuDW,
                                     void* newTaskGpuDW,
                                     void* stream,
                                     int deviceID) ) :
          ptr(ptr), pmf(pmf)
        {
        }
        virtual ~ActionDevice()
        {
        }

        //////////
        // Insert Documentation Here:
        virtual void doit(CallBackEvent event,
                          const ProcessorGroup* pg,
                          const PatchSubset* patches,
                          const MaterialSubset* matls,
                          DataWarehouse* fromDW,
                          DataWarehouse* toDW,
                          void* oldTaskGpuDW,
                          void* newTaskGpuDW,
                          void* stream,
                          int deviceID)
        {
          (ptr->*pmf)(event, pg, patches, matls, fromDW, toDW, oldTaskGpuDW, newTaskGpuDW, stream, deviceID);
        }
    };  // end class ActionDevice

    template<class T, class Arg1>
    class ActionDevice1 : public ActionBase {
        T* ptr;
        void (T::*pmf)(CallBackEvent event,
                        const ProcessorGroup* pg,
                        const PatchSubset* patches,
                        const MaterialSubset* matls,
                              DataWarehouse* fromDW,
                              DataWarehouse* toDW,
                              void* oldTaskGpuDW,
                              void* newTaskGpuDW,
                              void* stream,
                              int deviceID,
                              Arg1 arg1);
        Arg1 arg1;
      public:
        // class ActionDevice1
        ActionDevice1(T* ptr,
                      void (T::*pmf)(CallBackEvent event,
                                     const ProcessorGroup* pg,
                                     const PatchSubset* patches,
                                     const MaterialSubset* matls,
                                     DataWarehouse* fromDW,
                                     DataWarehouse* toDW,
                                     void* oldTaskGpuDW,
                                     void* newTaskGpuDW,
                                     void* stream,
                                     int deviceID,
                                     Arg1 arg1),
                      Arg1 arg1)
            : ptr(ptr), pmf(pmf), arg1(arg1)
        {
        }
        virtual ~ActionDevice1()
        {
        }

        //////////
        // Insert Documentation Here:
        virtual void doit(CallBackEvent event,
                          const ProcessorGroup* pg,
                          const PatchSubset* patches,
                          const MaterialSubset* matls,
                          DataWarehouse* fromDW,
                          DataWarehouse* toDW,
                          void* oldTaskGpuDW,
                          void* newTaskGpuDW,
                          void* stream,
                          int deviceID)
        {
          (ptr->*pmf)(event, pg, patches, matls, fromDW, toDW, oldTaskGpuDW, newTaskGpuDW, stream, deviceID, arg1);
        }
    };  // end class ActionDevice1

    template<class T, class Arg1, class Arg2>
    class ActionDevice2 : public ActionBase {
        T* ptr;
        void (T::*pmf)(CallBackEvent event,
                       const ProcessorGroup* pg,
                       const PatchSubset* patches,
                       const MaterialSubset* matls,
                       DataWarehouse* fromDW,
                       DataWarehouse* toDW,
                       void* oldTaskGpuDW,
                       void* newTaskGpuDW,
                       void* stream,
                       int deviceID,
                       Arg1 arg1,
                       Arg2 arg2);
        Arg1 arg1;
        Arg2 arg2;
      public:
        // class ActionDevice2
        ActionDevice2(T* ptr,
                      void (T::*pmf)(CallBackEvent event,
                                     const ProcessorGroup* pg,
                                     const PatchSubset* patches,
                                     const MaterialSubset* matls,
                                     DataWarehouse* fromDW,
                                     DataWarehouse* toDW,
                                     void* oldTaskGpuDW,
                                     void* newTaskGpuDW,
                                     void* stream,
                                     int deviceID,
                                     Arg1 arg1,
                                     Arg2 arg2),
                      Arg1 arg1,
                      Arg2 arg2)
            : ptr(ptr), pmf(pmf), arg1(arg1), arg2(arg2)
        {
        }
        virtual ~ActionDevice2()
        {
        }

        //////////
        // Insert Documentation Here:
        virtual void doit(CallBackEvent event,
                          const ProcessorGroup* pg,
                          const PatchSubset* patches,
                          const MaterialSubset* matls,
                          DataWarehouse* fromDW,
                          DataWarehouse* toDW,
                          void* oldTaskGpuDW,
                          void* newTaskGpuDW,
                          void* stream,
                          int deviceID)
        {
          (ptr->*pmf)(event, pg, patches, matls, fromDW, toDW, oldTaskGpuDW, newTaskGpuDW, stream, deviceID, arg1, arg2);
        }
    };  // end class ActionDevice2

    template<class T, class Arg1, class Arg2, class Arg3>
    class ActionDevice3 : public ActionBase {
        T* ptr;
        void (T::*pmf)(CallBackEvent event,
                       const ProcessorGroup* pg,
                       const PatchSubset* patches,
                       const MaterialSubset* matls,
                       DataWarehouse* fromDW,
                       DataWarehouse* toDW,
                       void* oldTaskGpuDW,
                       void* newTaskGpuDW,
                       void* stream,
                       int deviceID,
                       Arg1 arg1,
                       Arg2 arg2,
                       Arg3 arg3);
        Arg1 arg1;
        Arg2 arg2;
        Arg3 arg3;

      public:
        // class ActionDevice3
        ActionDevice3(T* ptr,
                      void (T::*pmf)(CallBackEvent event,
                                     const ProcessorGroup* pg,
                                     const PatchSubset* patches,
                                     const MaterialSubset* matls,
                                     DataWarehouse* fromDW,
                                     DataWarehouse* toDW,
                                     void* oldTaskGpuDW,
                                     void* newTaskGpuDW,
                                     void* stream,
                                     int deviceID,
                                     Arg1 arg1,
                                     Arg2 arg2,
                                     Arg3 arg3),
                      Arg1 arg1,
                      Arg2 arg2,
                      Arg3 arg3)
            : ptr(ptr), pmf(pmf), arg1(arg1), arg2(arg2), arg3(arg3)
        {
        }
        virtual ~ActionDevice3()
        {
        }

        //////////
        // Insert Documentation Here:
        virtual void doit(       CallBackEvent    event,
                           const ProcessorGroup * pg,
                           const PatchSubset    * patches,
                           const MaterialSubset * matls,
                                 DataWarehouse  * fromDW,
                                 DataWarehouse  * toDW,
                                 void* oldTaskGpuDW,
                                 void* newTaskGpuDW,
                                 void           * stream,
                                 int              deviceID)
        {
          (ptr->*pmf)(event, pg, patches, matls, fromDW, toDW, oldTaskGpuDW, newTaskGpuDW, stream, deviceID, arg1, arg2, arg3);
        }
    };  // end class ActionDevice3

    template<class T, class Arg1, class Arg2, class Arg3, class Arg4>
    class ActionDevice4 : public ActionBase {
        T* ptr;
        void (T::*pmf)(CallBackEvent event,
                       const ProcessorGroup* pg,
                       const PatchSubset* patches,
                       const MaterialSubset* matls,
                       DataWarehouse* fromDW,
                       DataWarehouse* toDW,
                       void* oldTaskGpuDW,
                       void* newTaskGpuDW,
                       void* stream,
                       int deviceID,
                       Arg1 arg1,
                       Arg2 arg2,
                       Arg3 arg3,
                       Arg4 arg4);
        Arg1 arg1;
        Arg2 arg2;
        Arg3 arg3;
        Arg4 arg4;
      public:
        // class ActionDevice4
        ActionDevice4(T* ptr,
                      void (T::*pmf)(CallBackEvent event,
                                     const ProcessorGroup* pg,
                                     const PatchSubset* patches,
                                     const MaterialSubset* matls,
                                     DataWarehouse* fromDW,
                                     DataWarehouse* toDW,
                                     void* oldTaskGpuDW,
                                     void* newTaskGpuDW,
                                     void * stream,
                                     int deviceID,
                                     Arg1 arg1,
                                     Arg2 arg2,
                                     Arg3 arg3,
                                     Arg4 arg4),
                      Arg1 arg1,
                      Arg2 arg2,
                      Arg3 arg3,
                      Arg4 arg4)
            : ptr(ptr), pmf(pmf), arg1(arg1), arg2(arg2), arg3(arg3), arg4(arg4)
        {
        }
        virtual ~ActionDevice4()
        {
        }

        //////////
        // Insert Documentation Here:
        virtual void doit(CallBackEvent event,
                          const ProcessorGroup* pg,
                          const PatchSubset* patches,
                          const MaterialSubset* matls,
                          DataWarehouse* fromDW,
                          DataWarehouse* toDW,
                          void* oldTaskGpuDW,
                          void* newTaskGpuDW,
                          void* stream,
                          int deviceID)
        {
          (ptr->*pmf)(event, pg, patches, matls, fromDW, toDW, oldTaskGpuDW, newTaskGpuDW, stream, deviceID, arg1, arg2, arg3, arg4);
        }
    };  // end class ActionDevice4

    template<class T, class Arg1, class Arg2, class Arg3, class Arg4, class Arg5>
    class ActionDevice5 : public ActionBase {
        T* ptr;
        void (T::*pmf)(CallBackEvent event,
                       const ProcessorGroup* pg,
                       const PatchSubset* patches,
                       const MaterialSubset* matls,
                       DataWarehouse* fromDW,
                       DataWarehouse* toDW,
                       void* oldTaskGpuDW,
                       void* newTaskGpuDW,
                       void* stream,
                       int deviceID,
                       Arg1 arg1,
                       Arg2 arg2,
                       Arg3 arg3,
                       Arg4 arg4,
                       Arg5 arg5);
        Arg1 arg1;
        Arg2 arg2;
        Arg3 arg3;
        Arg4 arg4;
        Arg5 arg5;
      public:
        // class ActionDevice5
        ActionDevice5( T* ptr,
                       void (T::*pmf)(CallBackEvent event,
                                      const ProcessorGroup* pg,
                                      const PatchSubset* patches,
                                      const MaterialSubset* matls,
                                      DataWarehouse * fromDW,
                                      DataWarehouse * toDW,
                                      void* oldTaskGpuDW,
                                      void* newTaskGpuDW,
                                      void* stream,
                                      int deviceID,
                                      Arg1 arg1,
                                      Arg2 arg2,
                                      Arg3 arg3,
                                      Arg4 arg4,
                                      Arg5 arg5),
                      Arg1 arg1,
                      Arg2 arg2,
                      Arg3 arg3,
                      Arg4 arg4,
                      Arg5 arg5)
            : ptr(ptr), pmf(pmf), arg1(arg1), arg2(arg2), arg3(arg3), arg4(arg4), arg5(arg5)
        {
        }
        virtual ~ActionDevice5()
        {
        }

        //////////
        // Insert Documentation Here:
        virtual void doit(CallBackEvent event,
                          const ProcessorGroup* pg,
                          const PatchSubset* patches,
                          const MaterialSubset* matls,
                          DataWarehouse* fromDW,
                          DataWarehouse* toDW,
                          void* oldTaskGpuDW,
                          void* newTaskGpuDW,
                          void* stream,
                          int deviceID)
        {
          (ptr->*pmf)(event, pg, patches, matls, fromDW, toDW, oldTaskGpuDW, newTaskGpuDW, stream, deviceID, arg1, arg2, arg3, arg4, arg5);
        }
    };  // end class ActionDevice5
    // end Device Action constructors

  public:
    // class Task

    enum WhichDW {
      OldDW = 0,
      NewDW = 1,
      CoarseOldDW = 2,
      CoarseNewDW = 3,
      ParentOldDW = 4,
      ParentNewDW = 5,
      TotalDWs = 6
    };

    enum {
      NoDW = -1,
      InvalidDW = -2
    };

    enum TaskType {
      Normal,
      Reduction,
      InitialSend,
      OncePerProc,  // make sure to pass a PerProcessorPatchSet to the addTask function
      Output,
      Spatial       // e.g. Radiometer task (spatial scheduling); must call task->setType(Task::Spatial)
    };  
    


    Task(const std::string& taskName, TaskType type)
        : d_taskName(taskName), d_action(0)
    {
      d_tasktype = type;
      initialize();
    }

    // begin CPU only Task constructors
    template<class T>
    Task(const std::string& taskName,
         T* ptr,
         void (T::*pmf)(const ProcessorGroup* pg,
                        const PatchSubset* patches,
                        const MaterialSubset* matls,
                        DataWarehouse* fromDW,
                        DataWarehouse* toDW))
        : d_taskName(taskName), d_action(new Action<T>(ptr, pmf))
    {
      d_tasktype = Normal;
      initialize();
    }

    template<class T, class Arg1>
    Task(const std::string& taskName,
         T* ptr,
         void (T::*pmf)(const ProcessorGroup* pg,
                        const PatchSubset* patches,
                        const MaterialSubset* matls,
                        DataWarehouse* fromDW,
                        DataWarehouse* toDW,
                        Arg1 arg1),
         Arg1 arg1)
        : d_taskName(taskName), d_action(new Action1<T, Arg1>(ptr, pmf, arg1))
    {
      d_tasktype = Normal;
      initialize();
    }

    template<class T, class Arg1, class Arg2>
    Task(const std::string& taskName,
         T* ptr,
         void (T::*pmf)(const ProcessorGroup* pg,
                        const PatchSubset* patches,
                        const MaterialSubset* matls,
                        DataWarehouse* fromDW,
                        DataWarehouse* toDW,
                        Arg1,
                        Arg2),
         Arg1 arg1,
         Arg2 arg2)
        :
          d_taskName(taskName),
            d_action(new Action2<T, Arg1, Arg2>(ptr, pmf, arg1, arg2))
    {
      d_tasktype = Normal;
      initialize();
    }

    template<class T, class Arg1, class Arg2, class Arg3>
    Task(const std::string& taskName,
         T* ptr,
         void (T::*pmf)(const ProcessorGroup* pg,
                        const PatchSubset* patches,
                        const MaterialSubset* matls,
                        DataWarehouse* fromDW,
                        DataWarehouse* toDW,
                        Arg1,
                        Arg2,
                        Arg3),
         Arg1 arg1,
         Arg2 arg2,
         Arg3 arg3)
        :
          d_taskName(taskName),
            d_action(new Action3<T, Arg1, Arg2, Arg3>(ptr, pmf, arg1, arg2, arg3))
    {
      d_tasktype = Normal;
      initialize();
    }

    template<class T, class Arg1, class Arg2, class Arg3, class Arg4>
    Task(const std::string& taskName,
         T* ptr,
         void (T::*pmf)(const ProcessorGroup* pg,
                        const PatchSubset* patches,
                        const MaterialSubset* matls,
                        DataWarehouse* fromDW,
                        DataWarehouse* toDW,
                        Arg1,
                        Arg2,
                        Arg3,
                        Arg4),
         Arg1 arg1,
         Arg2 arg2,
         Arg3 arg3,
         Arg4 arg4)
        :
          d_taskName(taskName),
            d_action(new Action4<T, Arg1, Arg2, Arg3, Arg4>(ptr, pmf, arg1, arg2, arg3, arg4))
    {
      d_tasktype = Normal;
      initialize();
    }

    template<class T, class Arg1, class Arg2, class Arg3, class Arg4, class Arg5>
    Task(const std::string& taskName,
         T* ptr,
         void (T::*pmf)(const ProcessorGroup* pg,
                        const PatchSubset* patches,
                        const MaterialSubset* matls,
                        DataWarehouse* fromDW,
                        DataWarehouse* toDW,
                        Arg1,
                        Arg2,
                        Arg3,
                        Arg4,
                        Arg5),
         Arg1 arg1,
         Arg2 arg2,
         Arg3 arg3,
         Arg4 arg4,
         Arg5 arg5)
        :
          d_taskName(taskName),
            d_action(new Action5<T, Arg1, Arg2, Arg3, Arg4, Arg5>(ptr, pmf, arg1, arg2, arg3, arg4, arg5))
    {
      d_tasktype = Normal;
      initialize();
    }
    // end CPU only Task constructors



    // begin Device Task constructors
    template<class T>
    Task(
         const std::string& taskName,
         T* ptr,
         void (T::*pmf)(CallBackEvent event,
                        const ProcessorGroup* pg,
                        const PatchSubset* patches,
                        const MaterialSubset* matls,
                        DataWarehouse* fromDW,
                        DataWarehouse* toDW,
                        void* old_TaskGpuDW,
                        void* new_TaskGpuDW,
                        void* stream,
                        int deviceID))
        :
          d_taskName(taskName),
            d_action(new ActionDevice<T>(ptr, pmf))
    {
      initialize();
      d_tasktype = Normal;
    }

    template<class T, class Arg1>
    Task(
         const std::string& taskName,
         T* ptr,
         void (T::*pmf)(CallBackEvent event,
                        const ProcessorGroup* pg,
                        const PatchSubset* patches,
                        const MaterialSubset* matls,
                        DataWarehouse* fromDW,
                        DataWarehouse* toDW,
                        void* oldTaskGpuDW,
                        void* newTaskGpuDW,
                        void* stream,
                        int deviceID,
                        Arg1 arg1),
         Arg1 arg1)
        :
          d_taskName(taskName),
            d_action(new ActionDevice1<T, Arg1>(ptr, pmf, arg1))
    {
      initialize();
      d_tasktype = Normal;
    }

    template<class T, class Arg1, class Arg2>
    Task(const std::string& taskName,
         T* ptr,
         void (T::*pmf)(CallBackEvent event,
                        const ProcessorGroup* pg,
                        const PatchSubset* patches,
                        const MaterialSubset* matls,
                        DataWarehouse* fromDW,
                        DataWarehouse* toDW,
                        void* oldTaskGpuDW,
                        void* newTaskGpuDW,
                        void* stream,
                        int deviceID,
                        Arg1 arg1,
                        Arg2 arg2),
         Arg1 arg1,
         Arg2 arg2)
        :
          d_taskName(taskName),
            d_action(new ActionDevice2<T, Arg1, Arg2>(ptr, pmf, arg1, arg2))
    {
      initialize();
      d_tasktype = Normal;
    }

    template<class T, class Arg1, class Arg2, class Arg3>
    Task(const std::string& taskName,
         T* ptr,
         void (T::*pmf)(CallBackEvent event,
                        const ProcessorGroup* pg,
                        const PatchSubset* patches,
                        const MaterialSubset* matls,
                        DataWarehouse* fromDW,
                        DataWarehouse* toDW,
                        void* oldTaskGpuDW,
                        void* newTaskGpuDW,
                        void* stream,
                        int deviceID,
                        Arg1 arg1,
                        Arg2 arg2,
                        Arg3 arg3),
         Arg1 arg1,
         Arg2 arg2,
         Arg3 arg3)
        :
          d_taskName(taskName),
            d_action(new ActionDevice3<T, Arg1, Arg2, Arg3>(ptr, pmf, arg1, arg2, arg3))
    {
      initialize();
      d_tasktype = Normal;
    }

    template<class T, class Arg1, class Arg2, class Arg3, class Arg4>
    Task(const std::string& taskName,
         T* ptr,
         void (T::*pmf)(CallBackEvent event,
                        const ProcessorGroup* pg,
                        const PatchSubset* patches,
                        const MaterialSubset* matls,
                        DataWarehouse* fromDW,
                        DataWarehouse* toDW,
                        void* oldTaskGpuDW,
                        void* newTaskGpuDW,
                        void* stream,
                        int deviceID,
                        Arg1 arg1,
                        Arg2 arg2,
                        Arg3 arg3,
                        Arg4 arg4),
         Arg1 arg1,
         Arg2 arg2,
         Arg3 arg3,
         Arg4 arg4)
        :
          d_taskName(taskName),
            d_action(new ActionDevice4<T, Arg1, Arg2, Arg3, Arg4>(ptr, pmf, arg1, arg2, arg3, arg4))
    {
      initialize();
      d_tasktype = Normal;
    }

    template<class T, class Arg1, class Arg2, class Arg3, class Arg4, class Arg5>
    Task(const std::string& taskName,
         T* ptr,
         void (T::*pmf)(CallBackEvent event,
                        const ProcessorGroup* pg,
                        const PatchSubset* patches,
                        const MaterialSubset* matls,
                        DataWarehouse* fromDW,
                        DataWarehouse* toDW,
                        void* oldTaskGpuDW,
                        void* newTaskGpuDW,
                        void* stream,
                        int deviceID,
                        Arg1 arg1,
                        Arg2 arg2,
                        Arg3 arg3,
                        Arg4 arg4,
                        Arg5 arg5),
         Arg1 arg1,
         Arg2 arg2,
         Arg3 arg3,
         Arg4 arg4,
         Arg5 arg5)
        :
          d_taskName(taskName),
            d_action(new ActionDevice5<T, Arg1, Arg2, Arg3, Arg4, Arg5>(ptr, pmf, arg1, arg2, arg3, arg4, arg5))
    {
      initialize();
      d_tasktype = Normal;
    }
    // end Device Task constructors

    void initialize();

    virtual ~Task();

    void hasSubScheduler(bool state = true);
    inline bool getHasSubScheduler() const
    {
      return d_hasSubScheduler;
    }
    void usesMPI(bool state);
    inline bool usesMPI() const
    {
      return d_usesMPI;
    }
    void usesThreads(bool state);
    inline bool usesThreads() const
    {
      return d_usesThreads;
    }
    void usesDevice(bool state);
    inline bool usesDevice() const
    {
      return d_usesDevice;
    }

    //////////
    // Insert Documentation Here:
    void subpatchCapable(bool state = true);

    enum MaterialDomainSpec {
      NormalDomain,  // <- Normal/default setting
      OutOfDomain,   // <- Require things from all material 
    };

    enum PatchDomainSpec {
      ThisLevel,  // <- Normal/default setting
      CoarseLevel,  // <- AMR :  The data on the coarse level under the range of the fine patches (including extra cells or boundary layers)
      FineLevel,  // <- AMR :  The data on the fine level under the range of the coarse patches (including extra cells or boundary layers)
      OtherGridDomain  // for when we copy data to new grid after a regrid.
    };

   //////////
    // Most general case
    void requires(WhichDW,
                  const VarLabel*,
                  const PatchSubset* patches,
                  PatchDomainSpec patches_dom,
                  int level_offset,
                  const MaterialSubset* matls,
                  MaterialDomainSpec matls_dom,
                  Ghost::GhostType gtype,
                  int numGhostCells = 0,
                  bool oldTG = false);

    //////////
    // Like general case, level_offset is not specified
    void requires(WhichDW,
                  const VarLabel*,
                  const PatchSubset* patches,
                  PatchDomainSpec patches_dom,
                  const MaterialSubset* matls,
                  MaterialDomainSpec matls_dom,
                  Ghost::GhostType gtype,
                  int numGhostCells = 0,
                  bool oldTG = false);

    //////////
    // Insert Documentation Here:
    void requires(WhichDW,
                  const VarLabel*,
                  Ghost::GhostType gtype,
                  int numGhostCells = 0,
                  bool oldTG = false);

    //////////
    // Insert Documentation Here:
    void requires(WhichDW,
                  const VarLabel*,
                  const PatchSubset* patches,
                  const MaterialSubset* matls,
                  Ghost::GhostType gtype,
                  int numGhostCells = 0,
                  bool oldTG = false);

    //////////
    // Insert Documentation Here:
    void requires(WhichDW,
                  const VarLabel*,
                  const PatchSubset* patches,
                  Ghost::GhostType gtype,
                  int numGhostCells = 0,
                  bool oldTG = false);

    //////////
    // Insert Documentation Here:
    void requires(WhichDW,
                  const VarLabel*,
                  const MaterialSubset* matls,
                  Ghost::GhostType gtype,
                  int numGhostCells = 0,
                  bool oldTG = false);

    //////////
    // Insert Documentation Here: 
    void requires(WhichDW,
                  const VarLabel*,
                  const MaterialSubset* matls,
                  MaterialDomainSpec matls_dom,
                  Ghost::GhostType gtype,
                  int numGhostCells = 0,
                  bool oldTG = false);

    //////////
    // Requires only for reduction variables
    void requires(WhichDW,
                  const VarLabel*,
                  const Level* level = 0,
                  const MaterialSubset* matls = 0,
                  MaterialDomainSpec matls_dom = NormalDomain,
                  bool oldTG = false);

    //////////
    // Requires for reduction variables or perpatch veriables
    void requires(WhichDW,
                  const VarLabel*,
                  const MaterialSubset* matls,
                  bool oldTG = false);

    //////////
    // Requires only for perpatch variables
    void requires(WhichDW,
                  const VarLabel*,
                  const PatchSubset* patches,
                  const MaterialSubset* matls = 0);

    //////////
    // Most general case
    void computes(const VarLabel*,
                  const PatchSubset* patches,
                  PatchDomainSpec patches_domain,
                  const MaterialSubset* matls,
                  MaterialDomainSpec matls_domain);

    //////////
    // Insert Documentation Here:
    void computes(const VarLabel*,
                  const PatchSubset* patches = 0,
                  const MaterialSubset* matls = 0);

    //////////
    // Insert Documentation Here:
    void computes(const VarLabel*,
                  const MaterialSubset* matls);

    //////////
    // Insert Documentation Here:
    void computes(const VarLabel*,
                  const MaterialSubset* matls,
                  MaterialDomainSpec matls_domain);

    //////////
    // Insert Documentation Here:
    void computes(const VarLabel*,
                  const PatchSubset* patches,
                  PatchDomainSpec patches_domain);

    //////////
    // Insert Documentation Here:
    void computes(const VarLabel*,
                  const Level* level,
                  const MaterialSubset* matls = 0,
                  MaterialDomainSpec matls_domain = NormalDomain);

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
    void modifiesWithScratchGhost(const VarLabel*,
                  const PatchSubset* patches,
                  PatchDomainSpec patches_domain,
                  const MaterialSubset* matls,
                  MaterialDomainSpec matls_domain,
                  Ghost::GhostType gtype,
                  int numGhostCells,
                  bool oldTG = false);
  
    void computesWithScratchGhost(const VarLabel*,
                  const MaterialSubset* matls,
                  MaterialDomainSpec matls_domain,
                  Ghost::GhostType gtype,
                  int numGhostCells,
                  bool oldTG = false);
  
    //////////
    // Most general case
    void modifies(const VarLabel*,
                  const PatchSubset* patches,
                  PatchDomainSpec patches_domain,
                  const MaterialSubset* matls,
                  MaterialDomainSpec matls_domain,
                  bool oldTG = false);

    //////////
    // Insert Documentation Here:
    void modifies(const VarLabel*,
                  const PatchSubset* patches,
                  const MaterialSubset* matls,
                  bool oldTG = false);

    //////////
    // Insert Documentation Here:
    void modifies(const VarLabel*,
                  const MaterialSubset* matls,
                  bool oldTG = false);

    //////////
    // Insert Documentation Here:
    void modifies(const VarLabel*,
                  const MaterialSubset* matls,
                  MaterialDomainSpec matls_domain,
                  bool oldTG = false);

    //////////
    // Insert Documentation Here:
    void modifies(const VarLabel*,
                  bool oldTG = false);

    //////////
    // Modify reduction vars
    void modifies(const VarLabel*,
                  const Level* level,
                  const MaterialSubset* matls = 0,
                  MaterialDomainSpec matls_domain = NormalDomain,
                  bool oldTG = false);

    //////////
    // Tells the task to actually execute the function assigned to it.
    //
    virtual void doit(CallBackEvent event,
                      const ProcessorGroup* pg,
                      const PatchSubset*,
                      const MaterialSubset*,
                      std::vector<DataWarehouseP>& dws,
                      void* oldTaskGpuDW,
                      void* newTaskGpuDW,
                      void* stream,
                      int deviceID );

    inline const std::string& getName() const
    {
      return d_taskName;
    }

    inline const PatchSet* getPatchSet() const
    {
      return patch_set;
    }

    inline const MaterialSet* getMaterialSet() const
    {
      return matl_set;
    }

    struct Edge;

    int d_phase;                    // synchronized phase id, for dynamic task scheduling
    int d_comm;                     // task communicator id, for threaded task scheduling


    int maxGhostCells;              // max ghost cells of this task
    int maxLevelOffset;             // max level offset of this task
    std::set<Task*> childTasks;
    std::set<Task*> allChildTasks;

    enum DepType {
      Modifies, Computes, Requires
    };

    struct Dependency {
        Dependency* next;
        DepType deptype;
        Task* task;
        const VarLabel* var;
        bool lookInOldTG;
        const PatchSubset* patches;
        const MaterialSubset* matls;
        const Level* reductionLevel;
        Edge* req_head;   // Used in compiling the task graph.
        Edge* req_tail;
        Edge* comp_head;
        Edge* comp_tail;
        PatchDomainSpec patches_dom;
        MaterialDomainSpec matls_dom;
        Ghost::GhostType gtype;
        WhichDW whichdw;  // Used only by Requires

        // in the multi-TG construct, this will signify that the required
        // var will be constructed by the old TG
        int numGhostCells;
        int level_offset;
        int mapDataWarehouse() const
        {
          return task->mapDataWarehouse(whichdw);
        }

        Dependency(DepType deptype,
                   Task* task,
                   WhichDW dw,
                   const VarLabel* var,
                   bool oldtg,
                   const PatchSubset* patches,
                   const MaterialSubset* matls,
                   PatchDomainSpec patches_dom = ThisLevel,
                   MaterialDomainSpec matls_dom = NormalDomain,
                   Ghost::GhostType gtype = Ghost::None,
                   int numGhostCells = 0,
                   int level_offset = 0);

        Dependency(DepType deptype,
                   Task* task,
                   WhichDW dw,
                   const VarLabel* var,
                   bool oldtg,
                   const Level* reductionLevel,
                   const MaterialSubset* matls,
                   MaterialDomainSpec matls_dom = NormalDomain);
        ~Dependency();
        inline void addComp(Edge* edge);
        inline void addReq(Edge* edge);

        constHandle<PatchSubset>
        getPatchesUnderDomain(const PatchSubset* domainPatches) const;

        constHandle<MaterialSubset>
        getMaterialsUnderDomain(const MaterialSubset* domainMaterials) const;

      private:
        static constHandle<PatchSubset>
        getOtherLevelPatchSubset(PatchDomainSpec dom,
                                 int level_offset,
                                 const PatchSubset* subset,
                                 const PatchSubset* domainSubset,
                                 int ngc);

        Dependency();
        Dependency& operator=(const Dependency& copy);
        Dependency(const Dependency&);
    };  // end struct Dependency

    struct Edge {
        const Dependency* comp;
        Edge* compNext;
        const Dependency* req;
        Edge* reqNext;
        inline Edge(const Dependency* comp,
                    const Dependency * req)
            : comp(comp), compNext(0), req(req), reqNext(0)
        {
        }
    };

    typedef std::multimap<const VarLabel*, Dependency*, VarLabel::Compare> DepMap;

    const Dependency* getComputes() const
    {
      return comp_head;
    }
    const Dependency* getRequires() const
    {
      return req_head;
    }
    const Dependency* getModifies() const
    {
      return mod_head;
    }

    Dependency* getComputes()
    {
      return comp_head;
    }
    Dependency* getRequires()
    {
      return req_head;
    }
    Dependency* getModifies()
    {
      return mod_head;
    }

    // finds if it computes or modifies var
    bool hasComputes(const VarLabel* var,
                     int matlIndex,
                     const Patch* patch) const;

    // finds if it requires or modifies var
    bool hasRequires(const VarLabel* var,
                     int matlIndex,
                     const Patch* patch,
                     Uintah::IntVector lowOffset,
                     Uintah::IntVector highOffset,
                     WhichDW dw) const;

    // finds if it modifies var
    bool hasModifies(const VarLabel* var,
                     int matlIndex,
                     const Patch* patch) const;

    bool isReductionTask() const
    {
      return d_tasktype == Reduction;
    }

    void setType(TaskType tasktype)
    {
      d_tasktype = tasktype;
    }
    TaskType getType() const
    {
      return d_tasktype;
    }

    //////////
    // Prints out information about the task...
    void display(std::ostream & out) const;

    //////////
    // Prints out all information about the task, including dependencies
    void displayAll(std::ostream & out) const;

    int mapDataWarehouse(WhichDW dw) const;
    DataWarehouse* mapDataWarehouse(WhichDW dw,
                                    std::vector<DataWarehouseP>& dws) const;

    int getSortedOrder() const
    {
      return sortedOrder;
    }

    void setSortedOrder(int order)
    {
      sortedOrder = order;
    }

    void setMapping(int dwmap[TotalDWs]);

    void setSets(const PatchSet* patches,
                 const MaterialSet* matls);

  private:
    // class Task
    Dependency* isInDepMap(const DepMap& depMap,
                           const VarLabel* var,
                           int matlIndex,
                           const Patch* patch) const;

    //////////
    // Insert Documentation Here:
    std::string d_taskName;

  protected:
    ActionBase* d_action;

  private:
    Dependency* comp_head;
    Dependency* comp_tail;
    Dependency* req_head;
    Dependency* req_tail;
    Dependency* mod_head;
    Dependency* mod_tail;

    DepMap d_requiresOldDW;
    DepMap d_computes;  // also contains modifies
    DepMap d_requires;  // also contains modifies
    DepMap d_modifies;

    const PatchSet* patch_set;
    const MaterialSet* matl_set;

    bool d_usesMPI;
    bool d_usesThreads;
    bool d_usesDevice;
    bool d_subpatchCapable;
    bool d_hasSubScheduler;
    TaskType d_tasktype;

    Task(const Task&);
    Task& operator=(const Task&);

    static const MaterialSubset* getGlobalMatlSubset();
    static MaterialSubset* globalMatlSubset;

    int dwmap[TotalDWs];
    int sortedOrder;

    friend std::ostream & operator <<(std::ostream & out,
                                      const Uintah::Task & task);
    friend std::ostream & operator <<(std::ostream & out,
                                      const Uintah::Task::TaskType & tt);
    friend std::ostream & operator <<(std::ostream & out,
                                      const Uintah::Task::Dependency & dep);

};
// end class Task

inline void Task::Dependency::addComp(Edge* edge)
{
  if (comp_tail)
    comp_tail->compNext = edge;
  else
    comp_head = edge;
  comp_tail = edge;
}
inline void Task::Dependency::addReq(Edge* edge)
{
  if (req_tail)
    req_tail->reqNext = edge;
  else
    req_head = edge;
  req_tail = edge;
}

}  // End namespace Uintah

// This must be at the bottom
#include <CCA/Ports/DataWarehouse.h>

#endif
