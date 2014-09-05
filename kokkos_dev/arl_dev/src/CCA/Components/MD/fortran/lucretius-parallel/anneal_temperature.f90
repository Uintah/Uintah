
subroutine anneal_temperature(Nsteps)
use temperature_anneal_data
use integrate_data, only : integration_step
use ensamble_data, only : temperature
implicit none
integer, intent(IN) :: Nsteps
real(8) ddd
ddd = dble(integration_step)/dble(Nsteps)
temperature = anneal_T%Tstart + (anneal_T%Tend-anneal_T%Tstart) * ddd
end subroutine anneal_temperature
