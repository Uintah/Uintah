#include "timer.hpp"
#include <algorithm>
#include <unistd.h>
using namespace std;
using namespace func_timer;
const char* timer_fout = "default_timing";
int timer_enabled = 0;
static FILE* fout;
//void __timer_init() __attribute__((constructor));
unsigned int setalarm (unsigned int seconds, unsigned long ms)
{
	struct itimerval iOld, iNew;
	iNew.it_interval.tv_usec = 0;
	iNew.it_interval.tv_sec = 0;
	iNew.it_value.tv_usec = (long int) ms;
	iNew.it_value.tv_sec = (long int) seconds;
	if (setitimer (ITIMER_REAL, &iNew, &iOld) < 0)
		return 0;
	else
		return iOld.it_value.tv_sec;
}
void __timer_init()
{
	static char fname[1024];
	snprintf(fname, sizeof(fname), "%s.%d", timer_fout, getpid());
	fout = fopen(timer_fout, "w");
}
extern bool disable_normal_allocator;
extern int UintahProfiler_allocator_ptr;
void handler(int sig)
{
	disable_normal_allocator = true;
	UintahProfiler_allocator_ptr = 0;
	if(!timer_enabled) return;
	int ii;
	void* tracePtrs[100];
	int count = backtrace( tracePtrs, 100 );
	//char** funcNames = backtrace_symbols( tracePtrs + 2, count - 2 );
		backtrace_symbols_fd(tracePtrs + 2, count - 2, fileno(fout));
	
	fputs("===\n", fout);
	/*for(ii = 2; ii < count; ii++ )
	{
    	//fprintf(fout, "%p", tracePtrs[ii]);
		//fputc(ii == count - 1?'\n':'\t', fout);
	}*/
	//free( funcNames );
	setalarm(0, 1000);
	fflush(fout);
	disable_normal_allocator = false;
}

timer_counter_t* func_timer::timer[MAX_TIMER_NUM];
unsigned func_timer::timer_count = 0;
inline const char* __timer_print_table__(const char* str,int width, FILE* fp)
{
    int len = strlen(str);
    int pl = len;
    if(len > width) len = len - width;
    else len = 0;
    for(int i = 0 ; i < pl - len ; i ++)
    {
        fputc(*str,fp);
        str ++;
    }
    for(int i = pl - len ; i < width ; i ++)
        fputc(' ',fp);
    fputc('|',fp);
    return str;
}
inline void __timer_print_header__(int width, FILE* fp)
{
    for(int i = 0 ; i < width; i ++ )
        fputc('-',fp);
    fputc('+',fp);
}
inline void __timer_print_row__(int N,int* widths,const char** strs, FILE* fp)
{
    while(true)
    {
        int flag = 0;
        for(int i = 0 ; i < N && !flag ; i ++)
            if(*strs[i] != 0) flag = 1;
        if(flag == 0) break;
        for(int i = 0 ; i < N ; i ++)
            strs[i] = __timer_print_table__(strs[i],widths[i], fp);
        fputc('\n',fp);
    }
    for(int i = 0 ; i < N ; i ++)
        __timer_print_header__(widths[i], fp);
    fputc('\n',fp);
}
bool __timer_compare__(const timer_counter_t* a,const timer_counter_t* b)
{
    int ret = strcmp(a->file,b->file);
    if(ret > 0) return false;
    if(ret < 0) return true;
    return a->line < b->line;
}
void timer_print_summary()
{
	FILE* fp = fopen("timing_summary.txt", "w");
    sort(timer,timer + timer_count,__timer_compare__);
    int N = 6;
    char buffer[10][10000] = {"        file        ","   line   "," function ","               statement                ","        count       ","        time        "};
    const char* ptr[10] = {buffer[0],buffer[1],buffer[2],buffer[3],buffer[4],buffer[5]};
    int widths[] = {20,10,10,40,20,20};
    for(int i = 0 ; i < N ; i ++)
        __timer_print_header__(widths[i], fp);
    fputc('\n',fp);
    __timer_print_row__(N,widths,ptr, fp);
    for(int i = 0 ; i < timer_count ; i ++)
    {
        snprintf(buffer[0],10000,"%s",timer[i]->file);
        snprintf(buffer[1],10000,"%10d",timer[i]->line);
        snprintf(buffer[2],10000,"%s",timer[i]->func);
        snprintf(buffer[3],10000,"%s",timer[i]->stmt);
        snprintf(buffer[4],10000,"%20llu",timer[i]->count);
        snprintf(buffer[5],10000,"%20llu",timer[i]->elapsed);
        const char* ptr[10] = {buffer[0],buffer[1],buffer[2],buffer[3],buffer[4],buffer[5]};
        __timer_print_row__(N,widths,ptr, fp);
    }
	fclose(fp);
}
