
      vk  = vele_G(ndx,i_pair)  ;  vk1 = vele_G(ndx+1,i_pair) ; vk2 = vele_G(ndx+2,i_pair)
       t1 = vk  + (vk1 - vk )*ppp
       t2 = vk1 + (vk2 - vk1)*(ppp - 1.0d0)
       B0 = (t1 + (t2-t1)*(ppp*0.5d0))
      vk  = gele_G(ndx,i_pair)  ;  vk1 = gele_G(ndx+1,i_pair) ; vk2 = gele_G(ndx+2,i_pair)
       t1 = vk  + (vk1 - vk )*ppp
       t2 = vk1 + (vk2 - vk1)*(ppp - 1.0d0)
       B1 = (t1 + (t2-t1)*(ppp*0.5d0))
      vk  = vele2_G(ndx,i_pair)  ;  vk1 = vele2_G(ndx+1,i_pair) ; vk2 = vele2_G(ndx+2,i_pair)
       t1 = vk  + (vk1 - vk )*ppp
       t2 = vk1 + (vk2 - vk1)*(ppp - 1.0d0)
       B2 = (t1 + (t2-t1)*(ppp*0.5d0))
      vk  = vele3_G(ndx,i_pair)  ;  vk1 = vele3_G(ndx+1,i_pair) ; vk2 = vele3_G(ndx+2,i_pair)
       t1 = vk  + (vk1 - vk )*ppp
       t2 = vk1 + (vk2 - vk1)*(ppp - 1.0d0)
       B3 = (t1 + (t2-t1)*(ppp*0.5d0))

