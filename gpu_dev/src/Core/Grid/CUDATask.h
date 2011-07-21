/*
 
 The MIT License
 
 Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
 Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
 University of Utah.
 
 License for the specific language governing rights and limitations under
 Permission is hereby granted, free of charge, to any person obtaining a 
 copy of this software and associated documentation files (the "Software"),
 to deal in the Software without restriction, including without limitation 
 the rights to use, copy, modify, merge, publish, distribute, sublicense, 
 and/or sell copies of the Software, and to permit persons to whom the 
 Software is furnished to do so, subject to the following conditions:
 
 The above copyright notice and this permission notice shall be included 
 in all copies or substantial portions of the Software.
 
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
 OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
 THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
 FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
 DEALINGS IN THE SOFTWARE.
 
 */

#ifdef HAVE_CUDA

#ifndef UINTAH_HOMEBREW_CUDATask_H
#define UINTAH_HOMEBREW_CUDATask_H

#include <Core/Exception/InternalError.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Ghost.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Util/constHandle.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Geometry/IntVector.h>
#include <CCA/Components/Schedules/CUDADevice.h>
#include <CCA/Ports/DataWarehouseP.h>

#include <map>
#include <set>
#include <vector>
#include <string>
#include <iostream>


#include <Core/Grid/uintahshare.h>
namespace Uintah {
    
    class Level;
    class DataWarehouse;
    class ProcessorGroup;
    
    /**************************************
     
     CLASS
     CUDATask
     
     A task to be run with CUDA.
     
     
     
     GENERAL INFORMATION
     
     CUDATask.h
     
     Alan P. Humphrey 
     Department of Computer Science
     University of Utah
     
     and
     
     Joseph R. Peterson
     Department of Chemistry
     University of Utah
     
     Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
     
     Copyright (C) 2011 SCI Group
     
     
     
     KEYWORDS
     Task CUDA GPU
     
     
     
     DESCRIPTION
     CUDATask is a specialized version of Task that have specialized 'Action' classes
     that pass the CUDA device to be used and the CUDADevice information.  
     
     
     WARNING
     
     ****************************************/
    
    class UINTAHSHARE CUDATask {

        class UINTAHSHARE CUDAActionBase : public ActionBase {
        public:
            virtual ~CUDAActionBase();
            virtual void doit(const ProcessorGroup* pc,
                              const PatchSubset* patches,
                              const MaterialSubset* matls,
                              DataWarehouse* fromDW,
                              DataWarehouse* toDW)
            {
                throw new InternalError("ERROR:CUDAActionBase: This function should never be called by a derived class of CUDAActionBase.  The version with the device id and CUDADevice parameters should be used instead." __FILE__, __LINE__); 
            }
            
            virtual void doit(const ProcessorGroup* pc,
                              const PatchSubset* patches,
                              const MaterialSubset* matls,
                              DataWarehouse* fromDW,
                              DataWarehouse* toDW,
                              int,
                              CUDADevice *) = 0;
        };
        
        template<class T>
        class CUDAAction : public CUDAActionBase {
            
            T* ptr;
            void (T::*pmf)(const ProcessorGroup*,
                           const PatchSubset* patches,
                           const MaterialSubset* matls,
                           DataWarehouse*,
                           DataWarehouse*,
                           int,
                           CUDADevice *);
        public: // class CUDAAction
            CUDAAction( T* ptr,
                   void (T::*pmf)(const ProcessorGroup*, 
                                  const PatchSubset* patches,
                                  const MaterialSubset* matls,
                                  DataWarehouse*,
                                  DataWarehouse*,
                                  int,
                                  CUDADevice *) )
            : ptr(ptr), pmf(pmf) {}
            virtual ~CUDAAction() {}
            
            //////////
            // Insert Documentation Here:
            virtual void doit(const ProcessorGroup* pc,
                              const PatchSubset* patches,
                              const MaterialSubset* matls,
                              DataWarehouse* fromDW,
                              DataWarehouse* toDW,
                              int dev,
                              CUDADevice *devprop) {
                (ptr->*pmf)(pc, patches, matls, fromDW, toDW, dev, devprop);
            }
        }; // end class CUDAAction
        
        template<class T, class Arg1>
        class CUDAAction1 : public CUDAActionBase {
            
            T* ptr;
            void (T::*pmf)(const ProcessorGroup*,
                           const PatchSubset* patches,
                           const MaterialSubset* matls,
                           DataWarehouse*,
                           DataWarehouse*,
                           int,
                           CUDADevice *,
                           Arg1 arg1);
            Arg1 arg1;
        public: // class CUDAAction1
            CUDAAction1( T* ptr,
                    void (T::*pmf)(const ProcessorGroup*, 
                                   const PatchSubset* patches,
                                   const MaterialSubset* matls,
                                   DataWarehouse*,
                                   DataWarehouse*,
                                   int,
                                   CUDADevice *,
                                   Arg1),
                    Arg1 arg1)
            : ptr(ptr), pmf(pmf), arg1(arg1) {}
            virtual ~CUDAAction1() {}
            
            //////////
            // Insert Documentation Here:
            virtual void doit(const ProcessorGroup* pc,
                              const PatchSubset* patches,
                              const MaterialSubset* matls,
                              DataWarehouse* fromDW,
                              DataWarehouse* toDW,
                              int dev,
                              CUDADevice *devprop) {
                (ptr->*pmf)(pc, patches, matls, fromDW, toDW, dev, devprop arg1);
            }
        }; // end class CUDAAction1
        
        template<class T, class Arg1, class Arg2>
        class CUDAAction2 : public CUDAActionBase {
            
            T* ptr;
            void (T::*pmf)(const ProcessorGroup*,
                           const PatchSubset* patches,
                           const MaterialSubset* matls,
                           DataWarehouse*,
                           DataWarehouse*,
                           int,
                           CUDADevice *,
                           Arg1 arg1, Arg2 arg2);
            Arg1 arg1;
            Arg2 arg2;
        public: // class CUDAAction2
            CUDAAction2( T* ptr,
                    void (T::*pmf)(const ProcessorGroup*, 
                                   const PatchSubset* patches,
                                   const MaterialSubset* matls,
                                   DataWarehouse*,
                                   DataWarehouse*,
                                   int,
                                   CUDADevice *,
                                   Arg1, Arg2),
                    Arg1 arg1, Arg2 arg2)
            : ptr(ptr), pmf(pmf), arg1(arg1), arg2(arg2) {}
            virtual ~CUDAAction2() {}
            
            //////////
            // Insert Documentation Here:
            virtual void doit(const ProcessorGroup* pc,
                              const PatchSubset* patches,
                              const MaterialSubset* matls,
                              DataWarehouse* fromDW,
                              DataWarehouse* toDW,
                              int dev,
                              CUDADevice *devprop) {
                (ptr->*pmf)(pc, patches, matls, fromDW, toDW, dev, devprop, arg1, arg2);
            }
        }; // end class CUDAAction2
        
        template<class T, class Arg1, class Arg2, class Arg3>
        class CUDAAction3 : pulbic CUDAActionBase {
            
            T* ptr;
            void (T::*pmf)(const ProcessorGroup*,
                           const PatchSubset* patches,
                           const MaterialSubset* matls,
                           DataWarehouse*,
                           DataWarehouse*,
                           int,
                           CUDADevice *,
                           Arg1 arg1, Arg2 arg2, Arg3 arg3);
            Arg1 arg1;
            Arg2 arg2;
            Arg3 arg3;
        public: // class CUDAAction3
            CUDAAction3( T* ptr,
                    void (T::*pmf)(const ProcessorGroup*, 
                                   const PatchSubset* patches,
                                   const MaterialSubset* matls,
                                   DataWarehouse*,
                                   DataWarehouse*,
                                   int,
                                   CUDADevice *,
                                   Arg1, Arg2, Arg3),
                    Arg1 arg1, Arg2 arg2, Arg3 arg3)
            : ptr(ptr), pmf(pmf), arg1(arg1), arg2(arg2), arg3(arg3) {}
            virtual ~CUDAAction3() {}
            
            //////////
            // Insert Documentation Here:
            virtual void doit(const ProcessorGroup* pc,
                              const PatchSubset* patches,
                              const MaterialSubset* matls,
                              DataWarehouse* fromDW,
                              DataWarehouse* toDW,
                              int dev,
                              CUDADevice *devprop) {
                (ptr->*pmf)(pc, patches, matls, fromDW, toDW, dev, devprop, arg1, arg2, arg3);
            }
        }; // end CUDAAction3
        
        template<class T, class Arg1, class Arg2, class Arg3, class Arg4>
        class CUDAAction4 : public CUDAActionBase {
            
            T* ptr;
            void (T::*pmf)(const ProcessorGroup*,
                           const PatchSubset* patches,
                           const MaterialSubset* matls,
                           DataWarehouse*,
                           DataWarehouse*,
                           int,
                           CUDADevice *,
                           Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4);
            Arg1 arg1;
            Arg2 arg2;
            Arg3 arg3;
            Arg4 arg4;
        public: // class CUDAAction4
            CUDAAction4( T* ptr,
                    void (T::*pmf)(const ProcessorGroup*, 
                                   const PatchSubset* patches,
                                   const MaterialSubset* matls,
                                   DataWarehouse*,
                                   DataWarehouse*,
                                   int,
                                   CUDADevice *,
                                   Arg1, Arg2, Arg3, Arg4),
                    Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4)
            : ptr(ptr), pmf(pmf), arg1(arg1), arg2(arg2),
            arg3(arg3), arg4(arg4) {}
            virtual ~CUDAAction4() {}
            
            //////////
            // Insert Documentation Here:
            virtual void doit(const ProcessorGroup* pc,
                              const PatchSubset* patches,
                              const MaterialSubset* matls,
                              DataWarehouse* fromDW,
                              DataWarehouse* toDW,
                              int dev,
                              CUDADevice *devprop) {
                (ptr->*pmf)(pc, patches, matls, fromDW, toDW, dev, devprop, arg1, arg2, arg3, arg4);
            }
        }; // end CUDAAction4
        
        template<class T, class Arg1, class Arg2, class Arg3, class Arg4, class Arg5>
        class CUDAAction5 : public CUDAActionBase {
            
            T* ptr;
            void (T::*pmf)(const ProcessorGroup*,
                           const PatchSubset* patches,
                           const MaterialSubset* matls,
                           DataWarehouse*,
                           DataWarehouse*,
                           int,
                           CUDADevice *,
                           Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5);
            Arg1 arg1;
            Arg2 arg2;
            Arg3 arg3;
            Arg4 arg4;
            Arg5 arg5;
        public: // class CUDAAction4
            CUDAAction5( T* ptr,
                    void (T::*pmf)(const ProcessorGroup*, 
                                   const PatchSubset* patches,
                                   const MaterialSubset* matls,
                                   DataWarehouse*,
                                   DataWarehouse*,
                                   int,
                                   CUDADevice *,
                                   Arg1, Arg2, Arg3, Arg4, Arg5),
                    Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5)
            : ptr(ptr), pmf(pmf), arg1(arg1), arg2(arg2),
            arg3(arg3), arg4(arg4), arg5(arg5) {}
            virtual ~CUDAAction5() {}
            
            //////////
            // Insert Documentation Here:
            virtual void doit(const ProcessorGroup* pc,
                              const PatchSubset* patches,
                              const MaterialSubset* matls,
                              DataWarehouse* fromDW,
                              DataWarehouse* toDW,
                              int dev,
                              CUDADevice *devprop) {
                (ptr->*pmf)(pc, patches, matls, fromDW, dev, devprop, toDW, arg1, arg2, arg3, arg4, arg5);
            }
        }; // end CUDAAction5
        
    public: // class CUDATask
        
        CUDATask(const std::string& taskName, TaskType type)
        :  Task(taskName, type)
        {
            this->setType(GPUCUDATask);
        }
        
        template<class T>
        CUDATask(const std::string&         taskName,
             T*                    ptr,
             void (T::*pmf)(const ProcessorGroup*,
                            const PatchSubset* patches,
                            const MaterialSubset* matls,
                            DataWarehouse*,
                            DataWarehouse*,
                            int,
                            CUDADevice *) )
        : Task(taskName, type), 
        d_action( scinew CUDAAction<T>(ptr, pmf) )
        {
            this->setType(GPUCUDATask);
        }
        
        template<class T, class Arg1>
        CUDATask(const std::string&         taskName,
             T*                    ptr,
             void (T::*pmf)(const ProcessorGroup*,
                            const PatchSubset* patches,
                            const MaterialSubset* matls,
                            DataWarehouse*,
                            DataWarehouse*,
                            int,
                            CUDADevice *,
                            Arg1),
             Arg1 arg1)
        : Taks(taskName, type), 
        d_action( scinew CUDAAction1<T, Arg1>(ptr, pmf, arg1) )
        {
            this->setType(GPUCUDATask);
        }
        
        template<class T, class Arg1, class Arg2>
        CUDATask(const std::string&         taskName,
             T*                    ptr,
             void (T::*pmf)(const ProcessorGroup*,
                            const PatchSubset* patches,
                            const MaterialSubset* matls,
                            DataWarehouse*,
                            DataWarehouse*,
                            int,
                            CUDADevice *,
                            Arg1, Arg2),
             Arg1 arg1, Arg2 arg2)
        : Task(taskName, type), 
        d_action( scinew CUDAAction2<T, Arg1, Arg2>(ptr, pmf, arg1, arg2) )
        {
            this->setType(GPUCUDATask);
        }
        
        template<class T, class Arg1, class Arg2, class Arg3>
        CUDATask(const std::string&         taskName,
             T*                    ptr,
             void (T::*pmf)(const ProcessorGroup*,
                            const PatchSubset* patches,
                            const MaterialSubset* matls,
                            DataWarehouse*,
                            DataWarehouse*,
                            int,
                            CUDADevice *,
                            Arg1, Arg2, Arg3),
             Arg1 arg1, Arg2 arg2, Arg3 arg3)
        : Task(taskName, type), 
        d_action( scinew CUDAAction3<T, Arg1, Arg2, Arg3>(ptr, pmf, arg1, arg2, arg3) )
        {
            this->setType(GPUCUDATask);
        }
        
        template<class T, class Arg1, class Arg2, class Arg3, class Arg4>
        CUDATask(const std::string&         taskName,
             T*                    ptr,
             void (T::*pmf)(const ProcessorGroup*,
                            const PatchSubset* patches,
                            const MaterialSubset* matls,
                            DataWarehouse*,
                            DataWarehouse*,
                            int,
                            CUDADevice *,
                            Arg1, Arg2, Arg3, Arg4),
             Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4)
        : Task(taskName, type), 
        d_action( scinew CUDAAction4<T, Arg1, Arg2, Arg3, Arg4>(ptr, pmf, arg1, arg2, arg3, arg4) )
        {
            this->setType(GPUCUDATask);
        }
        
        template<class T, class Arg1, class Arg2, class Arg3, class Arg4, class Arg5>
        CUDATask(const std::string&         taskName,
             T*                    ptr,
             void (T::*pmf)(const ProcessorGroup*,
                            const PatchSubset* patches,
                            const MaterialSubset* matls,
                            DataWarehouse*,
                            DataWarehouse*,
                            int,
                            CUDADevice *,
                            Arg1, Arg2, Arg3, Arg4, Arg5),
             Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5)
        : Task(tastName, type), 
        d_action( scinew CUDAAction5<T, Arg1, Arg2, Arg3, Arg4, Arg5>(ptr, pmf, arg1, arg2, arg3, arg4, arg5) )
        {
            this->setType(GPUCUDATask);
        }
        
        ~CUDATask();
        
        //////////
        // Tells the task to actually execute the function assigned to it.
        void doit(const ProcessorGroup* pc, const PatchSubset*,
                  const MaterialSubset*, vector<DataWarehouseP>& dws,
                  int, CUDADevice *);
        
    private: // class CUDATask
        CUDATask(const CUDATask&);
        CUDATask& operator=(const CUDATask&);
        
    }; // end class CUDATask
    
} // End namespace Uintah

// This must be at the bottom
#include <CCA/Ports/DataWarehouse.h>

#endif

#endif
