/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

/*
*
* Mathematical and Computational Sciences Division
* National Institute of Technology,
* Gaithersburg, MD USA
*
*
* This software was developed at the National Institute of Standards and
* Technology (NIST) by employees of the Federal Government in the course
* of their official duties. Pursuant to title 17 Section 105 of the
* United States Code, this software is not subject to copyright protection
* and is in the public domain.  NIST assumes no responsibility whatsoever for
* its use by other parties, and makes no guarantees, expressed or implied,
* about its quality, reliability, or any other characteristic.
*
*/



#ifndef STOPWATCH_H
#define STOPWATCH_H

// for clock() and CLOCKS_PER_SEC
#include <time.h>


namespace TNT
{

inline static double seconds(void)
{
    const double secs_per_tick = 1.0 / CLOCKS_PER_SEC;
    return ( (double) clock() ) * secs_per_tick;
}

class Stopwatch {
    private:
        int running_;
        double start_time_;
        double total_;

    public:
        Stopwatch();
        inline void start();
        inline double stop();
		inline double read();
		inline void resume();
		inline int running();
};

Stopwatch::Stopwatch() : running_(0), start_time_(0.0), total_(0.0) {}

void Stopwatch::start() 
{
	running_ = 1;
	total_ = 0.0;
	start_time_ = seconds();
}

double Stopwatch::stop()  
{
	if (running_) 
	{
         total_ += (seconds() - start_time_); 
         running_ = 0;
    }
    return total_; 
}

inline void Stopwatch::resume()
{
	if (!running_)
	{
		start_time_ = seconds();
		running_ = 1;
	}
}
		

inline double Stopwatch::read()   
{
	if (running_)
	{
		stop();
		resume();
	}
	return total_;
}


}
#endif
    

            
