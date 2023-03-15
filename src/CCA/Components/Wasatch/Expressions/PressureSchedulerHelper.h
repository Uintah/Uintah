/*
 * The MIT License
 *
 * Copyright (c) 2019-2023 The University of Utah
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

#include<vector>

#include <Core/Grid/Task.h>
#include <CCA/Ports/SolverInterface.h>
#include <Core/Grid/Variables/VarLabel.h>

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

    virtual void schedule_stage_1(bool& do_schedule_var, bool& use_new_dw_var, bool& is_first_solve_var) = 0;
    virtual void schedule_stage_2(bool& do_schedule_var, bool& use_new_dw_var, bool& is_first_solve_var) = 0;
    virtual void schedule_stage_3(bool& do_schedule_var, bool& use_new_dw_var, bool& is_first_solve_var) = 0;
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
    RKPressureSchedulerHelper(RKPressureSchedulerInterface * concrete_RK_scheduler, 
                              std::vector<bool> projection_control_parameters) 
                              : concrete_RK_scheduler_(nullptr),
                              d_i(projection_control_parameters)
                              {
        this->transition_to(concrete_RK_scheduler);
    }

    ~RKPressureSchedulerHelper(){
        delete concrete_RK_scheduler_;
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
    void schedule(bool& do_schedule_var, bool& use_new_dw_var, bool& is_first_solve_var,
                  const int RKStage ){
        switch (RKStage){
            case 1:
            this->concrete_RK_scheduler_->schedule_stage_1(do_schedule_var, use_new_dw_var, is_first_solve_var);
            break;
            case 2:
            this->concrete_RK_scheduler_->schedule_stage_2(do_schedule_var, use_new_dw_var, is_first_solve_var);
            break;
            case 3:
            this->concrete_RK_scheduler_->schedule_stage_3(do_schedule_var, use_new_dw_var, is_first_solve_var);
            break;
        }
        
    }
};

/**
 * @brief FE concrete scheduler classes
 * 
 */

class FEScheduler : public RKPressureSchedulerInterface{
    public:
    void schedule_stage_1(bool& do_schedule_var, bool& use_new_dw_var, bool& is_first_solve_var)
    {
        //scheduling stage 1
        do_schedule_var = true;
        use_new_dw_var = true;
        is_first_solve_var = true;
        
        this->RK_scheduler_helper_->transition_to(new FEScheduler);
    }

    void schedule_stage_2(bool& do_schedule_var, bool& use_new_dw_var, bool& is_first_solve_var)
    {
        // will never reach this function
    }
    void schedule_stage_3(bool& do_schedule_var, bool& use_new_dw_var, bool& is_first_solve_var)
    {
        // will never reach this function
    }
};



/**
 * @brief RK2 concrete scheduler classes
 * 
 */
class RK20Scheduler : public RKPressureSchedulerInterface{
    public:
    void schedule_stage_1(bool& do_schedule_var, bool& use_new_dw_var, bool& is_first_solve_var)
    {
        //no pressure solve at stage 1
        do_schedule_var = false;
        use_new_dw_var = false;
        is_first_solve_var = false;
    }

    void schedule_stage_2(bool& do_schedule_var, bool& use_new_dw_var, bool& is_first_solve_var)
    {
        //scheduling stage 2
        do_schedule_var = true;
        use_new_dw_var = true; // Uintah::Task::NewDW
        is_first_solve_var = true;
    }
    void schedule_stage_3(bool& do_schedule_var, bool& use_new_dw_var, bool& is_first_solve_var)
    {
        // will never reach this function
    }
};

class RK21Scheduler : public RKPressureSchedulerInterface{
    public:
    void schedule_stage_1(bool& do_schedule_var, bool& use_new_dw_var, bool& is_first_solve_var)
    {
        //scheduling stage 1
        do_schedule_var = true;
        use_new_dw_var = false; // Uintah::Task::OldDW
        is_first_solve_var = true;
    }

    void schedule_stage_2(bool& do_schedule_var, bool& use_new_dw_var, bool& is_first_solve_var)
    {
        //scheduling stage 2
        do_schedule_var = true;
        use_new_dw_var = true; // Uintah::Task::NewDW
        is_first_solve_var = false;
        
        if (this->RK_scheduler_helper_->d_i[0])
            this->RK_scheduler_helper_->transition_to(new RK21Scheduler);
        else this->RK_scheduler_helper_->transition_to(new RK20Scheduler);
    }
    void schedule_stage_3(bool& do_schedule_var, bool& use_new_dw_var, bool& is_first_solve_var)
    {
        // will never reach this function
    }
};

/**
 * @brief RK3 concrete scheduler classes
 * 
 */
class RK300Scheduler : public RKPressureSchedulerInterface{
    public:
    void schedule_stage_1(bool& do_schedule_var, bool& use_new_dw_var, bool& is_first_solve_var)
    {
        //no pressure solve at stage 1
        do_schedule_var = false;
        use_new_dw_var = false;
        is_first_solve_var = false;
    }

    void schedule_stage_2(bool& do_schedule_var, bool& use_new_dw_var, bool& is_first_solve_var)
    {
        //no pressure solve at stage 2
        do_schedule_var = false;
        use_new_dw_var = false;
        is_first_solve_var = false;
    }
    void schedule_stage_3(bool& do_schedule_var, bool& use_new_dw_var, bool& is_first_solve_var)
    {
        //scheduling stage 3
        do_schedule_var = true;
        use_new_dw_var = true; // Uintah::Task::NewDW
        is_first_solve_var = true;
    }
};

class RK310Scheduler : public RKPressureSchedulerInterface{
    public:
    void schedule_stage_1(bool& do_schedule_var, bool& use_new_dw_var, bool& is_first_solve_var)
    {
        //scheduling stage 1
        do_schedule_var = true;
        use_new_dw_var = false; // Uintah::Task::OldDW
        is_first_solve_var = true;
    }

    void schedule_stage_2(bool& do_schedule_var, bool& use_new_dw_var, bool& is_first_solve_var)
    {
        //no pressure solve at stage 2
        do_schedule_var = false;
        use_new_dw_var = true; // Uintah::Task::NewDW
        is_first_solve_var = false;
    }
    void schedule_stage_3(bool& do_schedule_var, bool& use_new_dw_var, bool& is_first_solve_var)
    {
        //scheduling stage 3
        do_schedule_var = true;
        use_new_dw_var = true; // Uintah::Task::NewDW
        is_first_solve_var = false;
    }
};

class RK301Scheduler : public RKPressureSchedulerInterface{
    public:
    void schedule_stage_1(bool& do_schedule_var, bool& use_new_dw_var, bool& is_first_solve_var)
    {
        //no pressure solve at stage 1
        do_schedule_var = false;
        use_new_dw_var = false; // Uintah::Task::OldDW
        is_first_solve_var = false;
    }

    void schedule_stage_2(bool& do_schedule_var, bool& use_new_dw_var, bool& is_first_solve_var)
    {
        //scheduling stage 2
        do_schedule_var = true;
        use_new_dw_var = true; // Uintah::Task::NewDW
        is_first_solve_var = true;
    }
    void schedule_stage_3(bool& do_schedule_var, bool& use_new_dw_var, bool& is_first_solve_var)
    {
        //scheduling stage 3
        do_schedule_var = true;
        use_new_dw_var = true; // Uintah::Task::NewDW
        is_first_solve_var = false;
    }
};

class RK311Scheduler : public RKPressureSchedulerInterface{
    public:
    void schedule_stage_1(bool& do_schedule_var, bool& use_new_dw_var, bool& is_first_solve_var)
    {
        //scheduling stage 1
        do_schedule_var = true;
        use_new_dw_var = false; // Uintah::Task::OldDW
        is_first_solve_var = true;
    }

    void schedule_stage_2(bool& do_schedule_var, bool& use_new_dw_var, bool& is_first_solve_var)
    {
        //scheduling stage 2
        do_schedule_var = true;
        use_new_dw_var = true; // Uintah::Task::NewDW
        is_first_solve_var = false;
    }
    void schedule_stage_3(bool& do_schedule_var, bool& use_new_dw_var, bool& is_first_solve_var)
    {
        //scheduling stage 3
        do_schedule_var = true;
        use_new_dw_var = true; // Uintah::Task::NewDW
        is_first_solve_var = false;

        if (!this->RK_scheduler_helper_->d_i[0] && !this->RK_scheduler_helper_->d_i[1])
            this->RK_scheduler_helper_->transition_to(new RK300Scheduler);
        else if (!this->RK_scheduler_helper_->d_i[0] && this->RK_scheduler_helper_->d_i[1])
            this->RK_scheduler_helper_->transition_to(new RK301Scheduler);
        else if (this->RK_scheduler_helper_->d_i[0] && !this->RK_scheduler_helper_->d_i[1])
            this->RK_scheduler_helper_->transition_to(new RK310Scheduler);
        else this->RK_scheduler_helper_->transition_to(new RK311Scheduler);
    }
};
#endif // PRESSURE_SCHEDULER_HELPER