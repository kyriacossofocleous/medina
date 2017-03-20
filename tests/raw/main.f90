
!*********************************************************************************
! Part of the MEDINA: MECCA - KPP Fortran to CUDA source-to-source pre-processor
!*********************************************************************************
!
!  This module is automatically generated by create_mz_kpp. Used as input for 
!  transformation. Chemistry is very simplified eg. "toy".
!
!  Author: Michail Alvanos
!



        PROGRAM MAIN

        USE messy_mecca_kpp

        ! kpp_integrate (time_step_len,Conc,ierrf,xNacc,xNrej,istatus,l_debug,PE) 
         IMPLICIT NONE
         PRINT *, "This is a test"
         !call myfunc(2)
         CALL kpp_integrate (1,99,0,1,2,3,4,5) 


        END PROGRAM MAIN

