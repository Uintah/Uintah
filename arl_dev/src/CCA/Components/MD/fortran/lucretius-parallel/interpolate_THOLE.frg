
      vk  = vele_THOLE(ndx,i_pair)  ;  vk1 = vele_THOLE(ndx+1,i_pair) ; vk2 = vele_THOLE(ndx+2,i_pair)
       t1 = vk  + (vk1 - vk )*ppp
       t2 = vk1 + (vk2 - vk1)*(ppp - 1.0d0)
       B0_THOLE = (t1 + (t2-t1)*(ppp*0.5d0))
      vk  = gele_THOLE(ndx,i_pair)  ;  vk1 = gele_THOLE(ndx+1,i_pair) ; vk2 = gele_THOLE(ndx+2,i_pair)
       t1 = vk  + (vk1 - vk )*ppp
       t2 = vk1 + (vk2 - vk1)*(ppp - 1.0d0)
       B1_THOLE = (t1 + (t2-t1)*(ppp*0.5d0))

