import numpy as np
from scipy.interpolate import CubicSpline as SciPyCubicSpline

class CubicSpline1D:
    def __init__(self, x, y):
        # Ensure x is strictly increasing
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
        # Compute cumulative arc length parameter s
        dx = np.diff(x)
        dy = np.diff(y)
        ds = np.hypot(dx, dy)
        self.s = np.concatenate(([0], np.cumsum(ds)))
        
        # Initialize separate splines for x and y as functions of s
        self.sx = CubicSpline1D(self.s, x)
        self.sy = CubicSpline1D(self.s, y)

    def calc_position(self, s):
        # Vectorized computation of positions
        return self.sx.calc_position(s), self.sy.calc_position(s)
    
    def calc_curvature(self, s):
        # Vectorized computation of curvature
        dx = self.sx.calc_first_derivative(s)
        ddx = self.sx.calc_second_derivative(s)
        dy = self.sy.calc_first_derivative(s)
        ddy = self.sy.calc_second_derivative(s)
        denominator = (dx**2 + dy**2)**1.5
        curvature = (ddy * dx - ddx * dy) / denominator
        return curvature

    def calc_third_derivative_curvature(self, s):
        # Vectorized computation of the third derivative of curvature
        dx = self.sx.calc_first_derivative(s)
        dy = self.sy.calc_first_derivative(s)
        ddx = self.sx.calc_second_derivative(s)
        ddy = self.sy.calc_second_derivative(s)
        dddx = self.sx.calc_third_derivative(s)
        dddy = self.sy.calc_third_derivative(s)
        
        numerator1 = dddy * dx - dddx * dy
        numerator2 = (ddy * dx - ddx * dy) * (dx * ddx + dy * ddy)
        denominator1 = (dx**2 + dy**2)**1.5
        denominator2 = (dx**2 + dy**2)**2.5
        dk = (numerator1 / denominator1) - 3 * (numerator2 / denominator2)
        return dk
        
    def calc_yaw(self, s):
        # Vectorized computation of yaw angle
        dx = self.sx.calc_first_derivative(s)
        dy = self.sy.calc_first_derivative(s)
        return np.arctan2(dy, dx)

def calc_spline_course(x, y, ds=0.01):
    # Initialize the 2D spline
    sp = CubicSpline2D(x, y)
    
    # Generate parameter s values
    s = np.arange(0, sp.s[-1], ds)
    
    # Vectorized computations
    rx, ry = sp.calc_position(s)
    ryaw = sp.calc_yaw(s)
    rk = sp.calc_curvature(s)
    
    return rx, ry, ryaw, rk, s
