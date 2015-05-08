#ifndef __TIMER__HPP__
#define __TIMER__HPP__
#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>
#include <execinfo.h>
void timer_print_summary();
extern int timer_enabled;
unsigned int setalarm (unsigned int seconds, unsigned long ms);
void handler(int sig);
namespace func_timer
{
    const unsigned MAX_TIMER_NUM = 10000;
    class timer_counter_t;
    extern timer_counter_t* timer[MAX_TIMER_NUM];
    extern unsigned timer_count;
    extern inline unsigned long long rdtsc()
    {
        unsigned h,l;
        asm volatile("rdtsc": "=a"(l),"=d"(h));
        return (((unsigned long long)h)<<32)|l;
    }
    class timer_counter_t
    {
    public:
        const char* file;
        const char* stmt;
        const char* func;
        unsigned    line;
        unsigned long long    elapsed;
        unsigned long long    count;
		static void timer_call(int signum)
		{
			timer_print_summary();
			exit(0);
		}

        timer_counter_t(const char* _file,const char* _stmt,const char* _func,int _line)
        {
            if(timer_count == 0)
            {
				signal(38, timer_call);
                atexit(timer_print_summary);
            }
            file = _file;
            stmt = _stmt;
            func = _func;
            line = _line;
            elapsed = 0;
            count = 0;
            if(timer_count < MAX_TIMER_NUM) 
                timer[timer_count++] = this;
            else
                fprintf(stderr,"Timer Warnning %s:%d we cannot install more timer\n",_file,_line);
        }
    };
    template <class T>
    inline T& aux_accumulate_runtime(T v,volatile timer_counter_t& timer,unsigned long long tvpre)
    {
        timer.elapsed += rdtsc() - tvpre;
        timer.count ++;
        return v;
    }
    class BlockWatcher
    {
    public:
        unsigned long long tv_beg;
        volatile timer_counter_t* tc;
        inline BlockWatcher(volatile timer_counter_t* _tc)
        {
            tv_beg = rdtsc();
            tc = _tc;
        }
        inline ~BlockWatcher()
        {
            tc->elapsed += rdtsc() - tv_beg;
            tc->count ++;
        }
    };
}
#ifndef TIMER_DISABLED
#define FuncTimer(expr) ({\
    volatile static func_timer::timer_counter_t __timer_counter_instance__(__FILE__,#expr,__FUNCTION__,__LINE__);\
    unsigned long long tv_begin = func_timer::rdtsc();\
    aux_accumulate_runtime(expr,__timer_counter_instance__,tv_begin);\
})
#define StatementTimer(stm) do{\
    volatile static func_timer::timer_counter_t __timer_counter_instance__(__FILE__,#stm,__FUNCTION__,__LINE__);\
    unsigned long long tv_begin = func_timer::rdtsc();\
    stm;\
    unsigned long long tv_end   = func_timer::rdtsc();\
    __timer_counter_instance__.elapsed += tv_end - tv_begin;\
    __timer_counter_instance__.count ++;\
}while(0)
#define BlockTimer(Name) \
    volatile static func_timer::timer_counter_t __timer_counter_instance__(__FILE__,"BlockTimer:"#Name,__FUNCTION__,__LINE__);\
    func_timer::BlockWatcher __block_watcher__(&__timer_counter_instance__);
#else
#define FuncTimer(expr) expr
#define StatementTimer(stm) stm
#define BlockTimer(Name) 
#endif
#endif
