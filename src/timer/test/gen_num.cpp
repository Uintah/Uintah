#include <stdio.h>
#include <timer.hpp>
extern int arr[1000000];
void gen_num()
{
    for(int i = 0 ; i < 1000000 ; i ++ )
        arr[i] = FuncTimer(rand());
}
