#include <stdio.h>
#include <timer.hpp>
#include <algorithm>
extern int arr[1000000];
void print_num()
{
    for(int i = 0 ; i < 1000000 ; i ++ )
        StatementTimer(printf("%d\n",FuncTimer(arr[i])));
}
