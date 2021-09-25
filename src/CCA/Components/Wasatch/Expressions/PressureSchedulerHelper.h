/*
 * The MIT License
 *
 * Copyright (c) 2019-2021 The University of Utah
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

/**
 *  \file   PressureSchedulerHelper.h
 *  \date   Sep, 2021
 *  \author Mokbel Karam
 */

#ifndef PRESSURE_SCHEDULER_HELPER
#define PRESSURE_SCHEDULER_HELPER

#include<iostream>
#include<vector>

#include <Core/Grid/Task.h>
#include <CCA/Ports/SolverInterface.h>
#include <Core/Grid/Variables/VarLabel.h>

namespace SchedulerHelper{
/**
 * @brief This Pressure scheduler helper implements the State design patterns
 * 
 */

class RKPressureSchedulerHelper;

/**
 * @brief This class is the interface to different RK pressure scheduler.
 *        Also provides a backreference to the RKPressureSchedulerHelper object, 
 *        associated with the concrete RKPressureScheduler. This backreference can be used 
 *        by the concrete RKPressureScheduler to transition the
 *        RKPressureSchedulerHelper to another concrete RKPressureScheduler.
 * 
 */
class RKPressureSchedulerInterface{
    protected:
    RKPressureSchedulerHelper * RK_scheduler_helper_;

    public:
    virtual ~RKPressureSchedulerInterface(){}

    void set_RK_scheduler_helper(RKPressureSchedulerHelper * RK_scheduler_helper){
        this->RK_scheduler_helper_=RK_scheduler_helper;
    }

    virtual void schedule_stage_1(const Uintah::LevelP& level,
                  Uintah::SchedulerP sched,
                  const Uintah::MaterialSet* const materials,
                  const int RKStage ) = 0;
    virtual void schedule_stage_2(const Uintah::LevelP& level,
                  Uintah::SchedulerP sched,
                  const Uintah::MaterialSet* const materials,
                  const int RKStage ) = 0;
    virtual void schedule_stage_3(const Uintah::LevelP& level,
                  Uintah::SchedulerP sched,
                  const Uintah::MaterialSet* const materials,
                  const int RKStage ) = 0;
};

/**
 * @brief The RKPressureSchedulerHelper defines the interface of interest to user. 
 *        It also maintains a reference to an instance of the concrete RKPressureScheduler, 
 *        which represents the current state of the RKPressureSchedulerHelper.
 * 
 */
class RKPressureSchedulerHelper{
    private:
    /**
    * @var concrete_RK_scheduler_ A reference to the current concrete RKPressureScheduler object.
    */
    RKPressureSchedulerInterface * concrete_RK_scheduler_;
    public:
    std::vector<bool> d_i;
    Uintah::SolverInterface& pressuerSolver;
    const Uintah::VarLabel* matrixLabel;
    const Uintah::VarLabel* pressureLabel;
    const Uintah::VarLabel* prhsLabel;
    RKPressureSchedulerHelper(RKPressureSchedulerInterface * concrete_RK_scheduler, 
                              std::vector<bool> projection_control_parameters,
                              Uintah::SolverInterface& solver,
                              const Uintah::VarLabel* matLabel,
                              const Uintah::VarLabel* pLabel,
                              const Uintah::VarLabel* prhsLabel) 
                              : concrete_RK_scheduler_(nullptr),
                              d_i(projection_control_parameters),
                              pressuerSolver(solver),
                              matrixLabel(matLabel),
                              pressureLabel(pLabel),
                              prhsLabel(prhsLabel){
        this->transition_to(concrete_RK_scheduler);
    }

    ~RKPressureSchedulerHelper(){
        delete concrete_RK_scheduler_;
        matrixLabel = nullptr;        // this class is not responsible of the distruction of matrixLabel
        pressureLabel = nullptr;      // this class is not responsible of the distruction of pressureLabel
        prhsLabel = nullptr;          // this class is not responsible of the distruction of prhsLabel
        }
    /**
     * @brief The RKPressureSchedulerHelper allows changing the concrete RKPressureScheduler object at runtime.
     * 
     * @param other_concrete_RK_scheduler 
     */
    void transition_to(RKPressureSchedulerInterface * other_concrete_RK_scheduler ){
        if (this->concrete_RK_scheduler_ != nullptr)
            delete this->concrete_RK_scheduler_;

        this->concrete_RK_scheduler_=other_concrete_RK_scheduler;
        this->concrete_RK_scheduler_->set_RK_scheduler_helper(this);
    }

    /**
     * @brief The RKPressureSchedulerHelper delegates part of its behavior to the current concrete RKPressureScheduler object.
     * 
     * @param level 
     * @param sched 
     * @param materials 
     * @param RKStage 
     */
    void schedule(const Uintah::LevelP& level,
                  Uintah::SchedulerP sched,
                  const Uintah::MaterialSet* const materials,
                  const int RKStage ){
        switch (RKStage){
            case 1:
            this->concrete_RK_scheduler_->schedule_stage_1(level,sched,materials,RKStage);
            break;
            case 2:
            this->concrete_RK_scheduler_->schedule_stage_2(level,sched,materials,RKStage);
            break;
            case 3:
            this->concrete_RK_scheduler_->schedule_stage_3(level,sched,materials,RKStage);
            break;
        }
        
    }
};

/**
 * @brief FE concrete scheduler classes
 * 
 */

class FE : public RKPressureSchedulerInterface{
    public:
    void schedule_stage_1(const Uintah::LevelP& level,
                  Uintah::SchedulerP sched,
                  const Uintah::MaterialSet* const materials,
                  const int RKStage ){
        std::cout<<"scheduling stage 1"<<std::endl;

        this->RK_scheduler_helper_->pressuerSolver.scheduleSolve( level, sched, materials, this->RK_scheduler_helper_->matrixLabel, Uintah::Task::NewDW,
                                                                  this->RK_scheduler_helper_->pressureLabel, true,
                                                                  this->RK_scheduler_helper_->prhsLabel, Uintah::Task::NewDW,
                                                                  this->RK_scheduler_helper_->pressureLabel, Uintah::Task::OldDW,
                                                                  true);
        this->RK_scheduler_helper_->transition_to(new FE);
    }

    void schedule_stage_2(const Uintah::LevelP& level,
                  Uintah::SchedulerP sched,
                  const Uintah::MaterialSet* const materials,
                  const int RKStage ){}
    void schedule_stage_3(const Uintah::LevelP& level,
                  Uintah::SchedulerP sched,
                  const Uintah::MaterialSet* const materials,
                  const int RKStage ){}
};



/**
 * @brief RK2 concrete scheduler classes
 * 
 */
class RK2_0 : public RKPressureSchedulerInterface{
    public:
    void schedule_stage_1(const Uintah::LevelP& level,
                  Uintah::SchedulerP sched,
                  const Uintah::MaterialSet* const materials,
                  const int RKStage ){
        std::cout<<"no pressure solve at stage 1"<<std::endl;
    }

    void schedule_stage_2(const Uintah::LevelP& level,
                  Uintah::SchedulerP sched,
                  const Uintah::MaterialSet* const materials,
                  const int RKStage ){
        std::cout<<"scheduling stage 2"<<std::endl;
        this->RK_scheduler_helper_->pressuerSolver.scheduleSolve( level, sched, materials, this->RK_scheduler_helper_->matrixLabel, Uintah::Task::NewDW,
                                                                  this->RK_scheduler_helper_->pressureLabel, true,
                                                                  this->RK_scheduler_helper_->prhsLabel, Uintah::Task::NewDW,
                                                                  this->RK_scheduler_helper_->pressureLabel, Uintah::Task::NewDW,
                                                                  true);
    }
    void schedule_stage_3(const Uintah::LevelP& level,
                  Uintah::SchedulerP sched,
                  const Uintah::MaterialSet* const materials,
                  const int RKStage ){}
};

class RK2_1 : public RKPressureSchedulerInterface{
    public:
    void schedule_stage_1(const Uintah::LevelP& level,
                  Uintah::SchedulerP sched,
                  const Uintah::MaterialSet* const materials,
                  const int RKStage ){
        std::cout<<"scheduling stage 1"<<std::endl;
        this->RK_scheduler_helper_->pressuerSolver.scheduleSolve( level, sched, materials, this->RK_scheduler_helper_->matrixLabel, Uintah::Task::NewDW,
                                                                  this->RK_scheduler_helper_->pressureLabel, true,
                                                                  this->RK_scheduler_helper_->prhsLabel, Uintah::Task::NewDW,
                                                                  this->RK_scheduler_helper_->pressureLabel, Uintah::Task::OldDW,
                                                                  true);
    }

    void schedule_stage_2(const Uintah::LevelP& level,
                  Uintah::SchedulerP sched,
                  const Uintah::MaterialSet* const materials,
                  const int RKStage ){
        std::cout<<"scheduling stage 2"<<std::endl;
        this->RK_scheduler_helper_->pressuerSolver.scheduleSolve( level, sched, materials, this->RK_scheduler_helper_->matrixLabel, Uintah::Task::NewDW,
                                                                  this->RK_scheduler_helper_->pressureLabel, true,
                                                                  this->RK_scheduler_helper_->prhsLabel, Uintah::Task::NewDW,
                                                                  this->RK_scheduler_helper_->pressureLabel, Uintah::Task::NewDW,
                                                                  false);
        if (this->RK_scheduler_helper_->d_i[0])
            this->RK_scheduler_helper_->transition_to(new RK2_1);
        else this->RK_scheduler_helper_->transition_to(new RK2_0);
    }
    void schedule_stage_3(const Uintah::LevelP& level,
                  Uintah::SchedulerP sched,
                  const Uintah::MaterialSet* const materials,
                  const int RKStage){}
};

/**
 * @brief RK3 concrete scheduler classes
 * 
 */
class RK3_00 : public RKPressureSchedulerInterface{
    public:
    void schedule_stage_1(const Uintah::LevelP& level,
                  Uintah::SchedulerP sched,
                  const Uintah::MaterialSet* const materials,
                  const int RKStage ){
        std::cout<<"no pressure solve at stage 1"<<std::endl;
    }

    void schedule_stage_2(const Uintah::LevelP& level,
                  Uintah::SchedulerP sched,
                  const Uintah::MaterialSet* const materials,
                  const int RKStage ){
        std::cout<<"no pressure solve at stage 2"<<std::endl;
    }
    void schedule_stage_3(const Uintah::LevelP& level,
                  Uintah::SchedulerP sched,
                  const Uintah::MaterialSet* const materials,
                  const int RKStage ){
        std::cout<<"scheduling stage 3"<<std::endl;
        this->RK_scheduler_helper_->pressuerSolver.scheduleSolve( level, sched, materials, this->RK_scheduler_helper_->matrixLabel, Uintah::Task::NewDW,
                                                                  this->RK_scheduler_helper_->pressureLabel, true,
                                                                  this->RK_scheduler_helper_->prhsLabel, Uintah::Task::NewDW,
                                                                  this->RK_scheduler_helper_->pressureLabel, Uintah::Task::NewDW,
                                                                  true);

    }
};

class RK3_10 : public RKPressureSchedulerInterface{
    public:
    void schedule_stage_1(const Uintah::LevelP& level,
                  Uintah::SchedulerP sched,
                  const Uintah::MaterialSet* const materials,
                  const int RKStage ){
        std::cout<<"scheduling stage 1"<<std::endl;
        this->RK_scheduler_helper_->pressuerSolver.scheduleSolve( level, sched, materials, this->RK_scheduler_helper_->matrixLabel, Uintah::Task::NewDW,
                                                                  this->RK_scheduler_helper_->pressureLabel, true,
                                                                  this->RK_scheduler_helper_->prhsLabel, Uintah::Task::NewDW,
                                                                  this->RK_scheduler_helper_->pressureLabel, Uintah::Task::OldDW,
                                                                  true);
    }

    void schedule_stage_2(const Uintah::LevelP& level,
                  Uintah::SchedulerP sched,
                  const Uintah::MaterialSet* const materials,
                  const int RKStage ){
        std::cout<<"no pressure solve at stage 2"<<std::endl;
    }
    void schedule_stage_3(const Uintah::LevelP& level,
                  Uintah::SchedulerP sched,
                  const Uintah::MaterialSet* const materials,
                  const int RKStage ){
        std::cout<<"scheduling stage 3"<<std::endl;
        this->RK_scheduler_helper_->pressuerSolver.scheduleSolve( level, sched, materials, this->RK_scheduler_helper_->matrixLabel, Uintah::Task::NewDW,
                                                                  this->RK_scheduler_helper_->pressureLabel, true,
                                                                  this->RK_scheduler_helper_->prhsLabel, Uintah::Task::NewDW,
                                                                  this->RK_scheduler_helper_->pressureLabel, Uintah::Task::NewDW,
                                                                  false);
    }
};

class RK3_01 : public RKPressureSchedulerInterface{
    public:
    void schedule_stage_1(const Uintah::LevelP& level,
                  Uintah::SchedulerP sched,
                  const Uintah::MaterialSet* const materials,
                  const int RKStage ){
        std::cout<<"no pressure solve at stage 1"<<std::endl;
    }

    void schedule_stage_2(const Uintah::LevelP& level,
                  Uintah::SchedulerP sched,
                  const Uintah::MaterialSet* const materials,
                  const int RKStage ){
        std::cout<<"scheduling stage 2"<<std::endl;
        this->RK_scheduler_helper_->pressuerSolver.scheduleSolve( level, sched, materials, this->RK_scheduler_helper_->matrixLabel, Uintah::Task::NewDW,
                                                                  this->RK_scheduler_helper_->pressureLabel, true,
                                                                  this->RK_scheduler_helper_->prhsLabel, Uintah::Task::NewDW,
                                                                  this->RK_scheduler_helper_->pressureLabel, Uintah::Task::NewDW,
                                                                  true);
    }
    void schedule_stage_3(const Uintah::LevelP& level,
                  Uintah::SchedulerP sched,
                  const Uintah::MaterialSet* const materials,
                  const int RKStage ){
        std::cout<<"scheduling stage 3"<<std::endl;
        this->RK_scheduler_helper_->pressuerSolver.scheduleSolve( level, sched, materials, this->RK_scheduler_helper_->matrixLabel, Uintah::Task::NewDW,
                                                                  this->RK_scheduler_helper_->pressureLabel, true,
                                                                  this->RK_scheduler_helper_->prhsLabel, Uintah::Task::NewDW,
                                                                  this->RK_scheduler_helper_->pressureLabel, Uintah::Task::NewDW,
                                                                  false);
    }
};

class RK3_11 : public RKPressureSchedulerInterface{
    public:
    void schedule_stage_1(const Uintah::LevelP& level,
                  Uintah::SchedulerP sched,
                  const Uintah::MaterialSet* const materials,
                  const int RKStage ){
        std::cout<<"scheduling stage 1"<<std::endl;

        this->RK_scheduler_helper_->pressuerSolver.scheduleSolve( level, sched, materials, this->RK_scheduler_helper_->matrixLabel, Uintah::Task::NewDW,
                                                                  this->RK_scheduler_helper_->pressureLabel, true,
                                                                  this->RK_scheduler_helper_->prhsLabel, Uintah::Task::NewDW,
                                                                  this->RK_scheduler_helper_->pressureLabel, Uintah::Task::OldDW,
                                                                  true);
    }

    void schedule_stage_2(const Uintah::LevelP& level,
                  Uintah::SchedulerP sched,
                  const Uintah::MaterialSet* const materials,
                  const int RKStage ){
        std::cout<<"scheduling stage 2"<<std::endl;
        this->RK_scheduler_helper_->pressuerSolver.scheduleSolve( level, sched, materials, this->RK_scheduler_helper_->matrixLabel, Uintah::Task::NewDW,
                                                                  this->RK_scheduler_helper_->pressureLabel, true,
                                                                  this->RK_scheduler_helper_->prhsLabel, Uintah::Task::NewDW,
                                                                  this->RK_scheduler_helper_->pressureLabel, Uintah::Task::NewDW,
                                                                  false);
    }
    void schedule_stage_3(const Uintah::LevelP& level,
                  Uintah::SchedulerP sched,
                  const Uintah::MaterialSet* const materials,
                  const int RKStage ){
        std::cout<<"scheduling stage 3"<<std::endl;
        this->RK_scheduler_helper_->pressuerSolver.scheduleSolve( level, sched, materials, this->RK_scheduler_helper_->matrixLabel, Uintah::Task::NewDW,
                                                                  this->RK_scheduler_helper_->pressureLabel, true,
                                                                  this->RK_scheduler_helper_->prhsLabel, Uintah::Task::NewDW,
                                                                  this->RK_scheduler_helper_->pressureLabel, Uintah::Task::NewDW,
                                                                  false);

        if (!this->RK_scheduler_helper_->d_i[0] && !this->RK_scheduler_helper_->d_i[1])
            this->RK_scheduler_helper_->transition_to(new RK3_00);
        else if (!this->RK_scheduler_helper_->d_i[0] && this->RK_scheduler_helper_->d_i[1])
            this->RK_scheduler_helper_->transition_to(new RK3_01);
        else if (this->RK_scheduler_helper_->d_i[0] && !this->RK_scheduler_helper_->d_i[1])
            this->RK_scheduler_helper_->transition_to(new RK3_10);
        else this->RK_scheduler_helper_->transition_to(new RK3_11);
    }
};
}//SchedulerHelper
#endif // PRESSURE_SCHEDULER_HELPER