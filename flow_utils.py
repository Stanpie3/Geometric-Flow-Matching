def CondOT_flow(x0, x1, t):
  xt = (1-t[:,None,None])*x0 + t[:,None,None]*x1
  return xt

def CondOT_ut(x0, x1, t):
  cond_ut = x1-x0
  return cond_ut