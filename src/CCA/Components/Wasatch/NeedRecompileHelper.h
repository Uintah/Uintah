/*
 * The MIT License
 *
 * Copyright (c) 2019-2026 The University of Utah
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
 *  \file   NeedRecompileHelper.h
 *  \date   Sep, 2021
 *  \author Mokbel Karam
 */

#ifndef NEED_RECOMPILE_HELPER
#define NEED_RECOMPILE_HELPER

#include <vector>

/**
 * @brief This header file implement a modified version of the Strategie design pattern
 *        
 *        RecompileConditionInterface declares the check_condition() function that is common 
 *        to all supported versions of the concrete RecompileCondition classes.
 *
 *        The NeedRecompileHelper uses this interface to call the function check_condition()
 *        defined by the Concrete RecompileCondition classes.
 */

class RecompileConditionInterface{

    public:
    virtual bool check_condition()=0;

    virtual ~RecompileConditionInterface(){}
};

/**
 * @brief The NeedRecompileHelper class maintains a reference to all the RecompileCondition
 *        objects. The NeedRecompileHelper does not know the concrete class of a 
 *        RecompileCondition. It should work with all strategies via the RecompileConditionInterface.
 * 
 */

class NeedRecompileHelper{

    private:

    std::vector<RecompileConditionInterface*> recompile_conditions_;

    public:
    NeedRecompileHelper(){}
    ~NeedRecompileHelper(){
        for ( std::vector<RecompileConditionInterface*>::iterator i = recompile_conditions_.begin(); i != recompile_conditions_.end(); ++i )
        delete *i;
    }
    /**
     * @brief add a reference to a new RecompileCondition object
     * 
     * @param recomp_cond a pointer to the RecompileCondition object
     */
    void add_condition(RecompileConditionInterface* recomp_cond){
        recompile_conditions_.push_back(recomp_cond);
    }

    /**
     * @brief This is the function called to determine the return value of the function needRecompile().
     *        It evaluates all the conditions added to NeedRecompileHelper object and determine the return value.
     * 
     * @return true, initiate the recompilation of the taskgraph if one of the condition objects returns true
     * @return false, otherwise.
     */
    bool check_conditions(){
        bool common_cond = false;
        for (int i =0; i < recompile_conditions_.size(); i++)
        {
            bool local_cond;
            local_cond = recompile_conditions_[i]->check_condition();
            if (local_cond) common_cond = true;
        }
        return common_cond;
    }
};
/**
 * @brief Concrete RecompileCondition classes
 * 
 */
class LowCostIntegratorNeedRecompile: public RecompileConditionInterface{
    /**
     * @brief This class define the condition for the recompilation of the taskgraph 
     *        when using Low-Cost integrators with incompressible flow.
     * 
     *        The Low-Cost integrators starts by using pressure projection on all the 
     *        the intermediate stages. Then after the first N timesteps the integrator
     *        will use pressure values from previous timesteps to construct approximations
     *        for the pressure at the intermidiate stages.
     * 
     *        The number N depends on how many previous values are needed.
     * 
     *        All the necessary information needed to determine the need for recompilation
     *        have to be passed through the constructor of this class.   
     * 
     */
    private:
    int timestep_threshold_;
    Uintah::timeStep_vartype& timeStep_;
    bool& low_cost_recompile_flag_;
    public:
    /**
     * @brief Construct a new Low Cost Integrator Need Recompile object
     * 
     * @param timeStep reference to the current timestep value.
     * @param timestep_threshold the timestep number when the recompilation needs to happen. 
     *        timestep_threshold is equivalent to N.
     * @param low_cost_recompile_flag reference to a flag that gets updated based on the condition result.
     * 
     */
    LowCostIntegratorNeedRecompile(Uintah::timeStep_vartype& timeStep, const int& timestep_threshold, bool& low_cost_recompile_flag):
    timeStep_(timeStep),
    timestep_threshold_(timestep_threshold),
    low_cost_recompile_flag_(low_cost_recompile_flag)
    {}
    ~LowCostIntegratorNeedRecompile(){}
    /**
     * @brief definition of the check_condition() function
     * 
     */
    bool check_condition()
    {
        if (timeStep_==timestep_threshold_? true:false)
        {
            low_cost_recompile_flag_ = true;
            return low_cost_recompile_flag_;
        }
        else return false;
    }
};

// =================================================================
// In case another condition is needed to activate the recompilation 
// of the task graph, create below a new class that implements the interface
// RecompileConditionInterface and add this class to the NeedRecompileHelper
// in Wasatch.cc.
// =================================================================
#endif // NEED_RECOMPILE_HELPER