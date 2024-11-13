import numpy as np
from scipy.interpolate import CubicSpline as SciPyCubicSpline

class CubicSpline1D:
    def __init__(self, x, y):
        h = np.diff(x)
        if np.any(h <= 0):
            raise ValueError("x coordinates must be strictly increasing")
        
        self.x = x
        self.y = y
        self.nx = len(x)
        
        # Initialize SciPy CubicSpline with natural boundary conditions
        self.spline = SciPyCubicSpline(x, y, bc_type='natural')
    
    def calc_position(self, x):
        if x < self.x[0] or x > self.x[-1]:
            return None
        return self.spline(x)

    def calc_first_derivative(self, x):
        if x < self.x[0] or x > self.x[-1]:
            return None
        return self.spline(x, 1)

    def calc_second_derivative(self, x):
        if x < self.x[0] or x > self.x[-1]:
            return None
        return self.spline(x, 2)

    def calc_third_derivative(self, x):
        if x < self.x[0] or x > self.x[-1]:
            return None
        return self.spline(x, 3)


class CubicSpline2D:
    def __init__(self, x, y):
        self.s = self.__calc_s(x, y)
        self.sx = CubicSpline1D(self.s, x)
        self.sy = CubicSpline1D(self.s, y)

    def __calc_s(self, x, y):
        dx = np.diff(x)
        dy = np.diff(y)
        ds = np.hypot(dx, dy)
        s = np.concatenate(([0], np.cumsum(ds)))
        return s

    def calc_position(self, s):
        return self.sx.calc_position(s), self.sy.calc_position(s)

    def calc_curvature(self, s):
        dx = self.sx.calc_first_derivative(s)
        ddx = self.sx.calc_second_derivative(s)
        dy = self.sy.calc_first_derivative(s)
        ddy = self.sy.calc_second_derivative(s)
        k = (ddy * dx - ddx * dy) / ((dx ** 2 + dy ** 2)**(3 / 2))
        return k

    def calc_third_derivative_curvature(self, s):
        dx = self.sx.calc_first_derivative(s)
        dy = self.sy.calc_first_derivative(s)
        ddx = self.sx.calc_second_derivative(s)
        ddy = self.sy.calc_second_derivative(s)
        dddx = self.sx.calc_third_derivative(s)
        dddy = self.sy.calc_third_derivative(s)
        
        # Calculating third derivative of curvature
        dk = (dddy * dx - dddx * dy) / ((dx ** 2 + dy ** 2)**(3 / 2)) - \
             3 * (ddy * dx - ddx * dy) * (dx * ddx + dy * ddy) / ((dx ** 2 + dy ** 2)**(5 / 2))
        
        return dk

    def calc_yaw(self, s):
        dx = self.sx.calc_first_derivative(s)
        dy = self.sy.calc_first_derivative(s)
        return np.arctan2(dy, dx)

def calc_spline_course(x, y, ds=0.1):
    sp = CubicSpline2D(x, y)
    s = np.arange(0, sp.s[-1], ds)
    rx, ry, ryaw, rk = [], [], [], []
    
    for i_s in s:
        ix, iy = sp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        ryaw.append(sp.calc_yaw(i_s))
        rk.append(sp.calc_curvature(i_s))

    return rx, ry, ryaw, rk, s