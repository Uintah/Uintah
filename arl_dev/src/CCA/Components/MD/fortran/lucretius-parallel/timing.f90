module timing_module

type all_time_type
 integer :: year
 integer :: month
 integer :: day
 integer :: hour
 integer :: min
 integer :: sec
end type all_time_type

 type (all_time_type) starting_date
 type (all_time_type) ending_date


 private :: get_final_date
 private :: get_time_diff
 private :: get_diff
 public :: measure_clock
 public :: get_time_diference
 contains

 subroutine measure_clock(date)
 implicit none
 type (all_time_type) , intent(OUT) :: date
 integer time(8)
   call date_and_time (values=time)
   date%year = time(1)
   date%month = time(2)
   date%day  = time(3)
   date%hour = time(5)
   date%min = time(6)
   date%sec = time(7)
 end subroutine measure_clock 

 subroutine get_time_diference(diff)
   implicit none
   real(8), intent(OUT) :: diff
   call measure_clock(ending_date)
   call get_diff(starting_date,ending_date,diff)
 end subroutine get_time_diference

 subroutine get_diff(date, date1, diff)
implicit none
real(8),intent(OUT):: diff
type(all_time_type),intent(IN):: date,date1
integer i,j,k 
real(8) suma, sum1,suma1
logical l_bissect
integer max_day_of_month(12),dday, dmonth
!  date1 latter date
! The difference is in hours

suma= dble(date%year)*365.0d0*24.0d0*3600.0d0 + &
      dble(date%month-1.0d0)*(365.0d0/12.0d0)*24.0d0*3600.0d0+&
      dble(date%day-1.0d0)*24.0d0*3600.0d0+ &
      dble(date%hour)*3600.0d0 + dble(date%min)*60.0d0 + dble(date%sec) 

suma1 = dble(date1%year)*365.0d0*24.0d0*3600.0d0 + &
      dble(date1%month-1.0d0)*(365.0d0/12.0d0)*24.0d0*3600.0d0+&
      dble(date1%day-1.0d0)*24.0d0*3600.0d0+ &
      dble(date1%hour)*3600.0d0 + dble(date1%min)*60.0d0 + dble(date1%sec)


i=1
sum1 = suma1 - suma

diff = sum1/3600.0d0 ! diff is in hours
end subroutine get_diff




subroutine get_final_date(date, date1, diff)
implicit none
real(8),intent(IN) :: diff   ! diff is in sec
type(all_time_type),intent(IN) :: date
type(all_time_type),intent(OUT) ::date1
integer i,j,k 
real(8) suma, sum1
logical l_bissect
integer max_day_of_month(12),dday, dmonth

max_day_of_month = (/ 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 /)


suma= dble(date%year)*365.0d0*24.0d0*3600.0d0 + &
      dble(date%month-1.0d0)*(365.0d0/12.0d0)*24.0d0*3600.0d0+dble(date%day-1.0d0)*24.0d0*3600.0d0+ &
      dble(date%hour)*3600.0d0 + dble(date%min)*60.0d0 + dble(date%sec) 



i=1
sum1 = suma - diff


date1%year=INT(sum1/(365.0d0*24.0d0*3600.0d0))
sum1=sum1-dble(date1%year)*365.0d0*24.0d0*3600.0d0
!print*, 'year=',date1%year, sum1

date1%month = INT(sum1/(365/12.0d0*24.0d0*3600.0d0))    
sum1=sum1-dble(date1%month)*(365/12.0d0*24.0d0*3600.0d0)
date1%month = date1%month + 1
!print*, 'month=',date1%month, sum1

date1%day=INT(sum1/(24.0d0*3600.0d0))                   
sum1=sum1-dble(date1%day)*(24.0d0*3600.0d0)
date1%day = date1%day + 1
!print*, 'day=',date1%day, sum1

date1%hour=INT(sum1/3600.0d0) 
sum1=sum1-dble(date1%hour)*3600
!print*, 'hour=',date1%hour, sum1

date1%min=INT(sum1/60.0d0)
sum1=sum1-dble(date1%min)*60
!print*, 'min=',date1%min, sum1

date1%sec=INT(sum1)
!print*, 'sec=',date1%sec


l_bissect = (mod(date1%year,4) == 0)
if (l_bissect) max_day_of_month(2) = 29

if (date1%day > max_day_of_month(date1%month)) then
   date1%day=  (date1%day-max_day_of_month(date1%month))
   date1%month = date1%month + 1
endif


end subroutine get_final_date

end module timing_module
