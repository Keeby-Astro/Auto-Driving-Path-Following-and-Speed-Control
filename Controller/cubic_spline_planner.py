import numpy as np
from scipy.interpolate import CubicSpline as SciPyCubicSpline

class CubicSpline1D:
    def __init__(self, x, y):
        if np.any(np.diff(x) <= 0):
            raise ValueError("x coordinates must be strictly increasing")
        self.spline = SciPyCubicSpline(x, y, bc_type='natural')
        
    def calc_position(self, s):
        return self.spline(s)
        
    def calc_first_derivative(self, s):
        return self.spline(s, 1)
        
    def calc_second_derivative(self, s):
        return self.spline(s, 2)
        
    def calc_third_derivative(self, s):
        return self.spline(s, 3)
    
class CubicSpline2D:
    def __init__(self, x, y):
        dx = np.diff(x)
        dy = np.diff(y)
        ds = np.hypot(dx, dy)
        self.s = np.concatenate(([0], np.cumsum(ds)))
        self.sx = CubicSpline1D(self.s, x)
        self.sy = CubicSpline1D(self.s, y)
    
    def calc_position(self, s):
        return self.sx.calc_position(s), self.sy.calc_position(s)
    
def calc_spline_course(x, y, ds=0.01):
    sp = CubicSpline2D(x, y)
    s = np.arange(0, sp.s[-1], ds)
    rx, ry = sp.calc_position(s)
    dx = sp.sx.calc_first_derivative(s)
    dy = sp.sy.calc_first_derivative(s)
    ddx = sp.sx.calc_second_derivative(s)
    ddy = sp.sy.calc_second_derivative(s)
    ryaw = np.arctan2(dy, dx)
    rk = (ddy * dx - ddx * dy) / (dx**2 + dy**2)**1.5
    return rx, ry, ryaw, rk, s
